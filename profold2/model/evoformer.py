from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

from profold2.model.commons import *

# main evoformer class

class TemplateEmbedding(nn.Module):
    def __init__(self,
                 dim, 
                 heads = 8,
                 dim_head = 64,
                 max_seq_len = 2048,
                 attn_dropout = 0.,
                 templates_dim = 32,
                 templates_embed_layers = 4,
                 templates_angles_feats_dim = 55):
        super().__init__()

        self.to_template_embed = nn.Linear(templates_dim, dim)
        self.templates_embed_layers = templates_embed_layers
        self.template_pairwise_embedder = PairwiseAttentionBlock(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            seq_len = max_seq_len
        )

        self.template_pointwise_attn = Attention(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            dropout = attn_dropout
        )

        self.template_angle_mlp = nn.Sequential(
            nn.Linear(templates_angles_feats_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )   

    def forward(self, x, x_mask, m, msa_mask,
            templates_feats=None, templates_angles=None, templates_mask=None):
        # embed templates, if present
        if exists(templates_feats):
            assert exists(templates_mask)
            _, num_templates, *_ = templates_feats.shape

            # embed template

            t = self.to_template_embed(templates_feats)
            t_mask_crossed = rearrange(templates_mask, 'b t i -> b t i ()') * rearrange(templates_mask, 'b t j -> b t () j')

            t = rearrange(t, 'b t ... -> (b t) ...')
            t_mask_crossed = rearrange(t_mask_crossed, 'b t ... -> (b t) ...')

            for _ in range(self.templates_embed_layers):
                t = self.template_pairwise_embedder(t, mask = t_mask_crossed)

            t = rearrange(t, '(b t) ... -> b t ...', t = num_templates)
            t_mask_crossed = rearrange(t_mask_crossed, '(b t) ... -> b t ...', t = num_templates)

            # template pos emb

            x_point = rearrange(x, 'b i j d -> (b i j) () d')
            t_point = rearrange(t, 'b t i j d -> (b i j) t d')
            x_mask_point = rearrange(x_mask, 'b i j -> (b i j) ()')
            t_mask_point = rearrange(t_mask_crossed, 'b t i j -> (b i j) t')

            template_pooled = self.template_pointwise_attn(
                x_point,
                context = t_point,
                mask = x_mask_point,
                context_mask = t_mask_point
            )

            template_pooled_mask = rearrange(t_mask_point.sum(dim = -1) > 0, 'b -> b () ()')
            template_pooled = template_pooled * template_pooled_mask

            template_pooled = rearrange(template_pooled, '(b i j) () d -> b i j d', i = n, j = n)
            x += template_pooled

        # add template angle features to MSAs by passing through MLP and then concat
        if exists(templates_angles):
            assert exists(templates_mask)
            t_angle_feats = self.template_angle_mlp(templates_angles)
            m = torch.cat((m, t_angle_feats), dim = 1)
            msa_mask = torch.cat((msa_mask, templates_mask), dim = 1)
        return x, x_mask, m, msa_mask

class EvoformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        seq_len,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        global_column_attn = False
    ):
        super().__init__()
        self.layer = nn.ModuleList([
            PairwiseAttentionBlock(dim = dim, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout, global_column_attn = global_column_attn),
            FeedForward(dim = dim, dropout = ff_dropout),
            MsaAttentionBlock(dim = dim, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout),
            FeedForward(dim = dim, dropout = ff_dropout),
        ])

    def forward(self, inputs):
        x, m, mask, msa_mask = inputs
        attn, ff, msa_attn, msa_ff = self.layer

        # msa attention and transition

        m = msa_attn(m, mask = msa_mask, pairwise_repr = x)
        m = msa_ff(m) + m

        # pairwise attention and transition

        x = attn(x, mask = mask, msa_repr = m, msa_mask = msa_mask)
        x = ff(x) + x

        return x, m, mask, msa_mask

class Evoformer(nn.Module):
    def __init__(
        self,
        *,
        depth,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([EvoformerBlock(**kwargs) for _ in range(depth)])

    def forward(
        self,
        x,
        m,
        mask = None,
        msa_mask = None
    ):
        inp = (x, m, mask, msa_mask)
        x, m, *_ = checkpoint_sequential(self.layers, 1, inp)
        return x, m

