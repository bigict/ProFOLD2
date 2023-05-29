#!/bin/bash
#

CWD=`readlink -f $0`
CWD=`dirname ${CWD}`

cd ${CWD}

help() {
  echo "usage: `basename $0` -d <data_dir> -o <output_dir> -s -p <pf2_params> input_fasta ..."
  exit $1
}

create_sto() {
  if=$1
  of=$2
  cat ${if} | awk '
      BEGIN {
        print "# STOCKHOLM 1.0";
        print "#=GF SQ 1";
      } {
        if ($0~/^>/) {
          printf("%s ",substr($1, 2));
        } else {
          print $0;
        }
      } END {
        print "#=GS 7EDJ_1 AC 7EDJ_1";
        print "#=GS 7EDJ_1 DE 7EDJ_1";
        print "#=GC RF";
        print "//";
      }' > ${of}
}

docker_image_name="profold2"
data_dir=/mnt/data/profold2/db
output_dir=/mnt/data/profold2/test
max_template_date="2021-05-14"
msa_mode=1   # msa
pf2_params=""
while getopts 'd:o:m:t:p:sh' OPT; do
  case $OPT in
    m) docker_image_name="$OPTARG";;
    d) data_dir="$OPTARG";;
    o) output_dir="$OPTARG";;
    p) pf2_params="$OPTARG";;
    t) max_template_date="$OPTARG";;
    s) msa_mode=0;;
    h) help 0;;
    ?) help 1;;
  esac
done
shift $((OPTIND - 1))

mkdir -p ${output_dir}
for f in $*; do
  echo "Start predict (${f}|${msa_mode}) `date`"
  if [ ${msa_mode} -eq 0 ]; then
    fasta_name=$(basename ${f}|sed 's/\.fasta$//g')
    msa_dir="${output_dir}/${fasta_name}/msas"
    mkdir -p ${msa_dir}
    uniref90_hits_sto="${msa_dir}/uniref90_hits.sto"
    if [ ! -e ${uniref90_hits_sto} ]; then
      create_sto ${f} ${uniref90_hits_sto}
    fi
    mgnify_hits_sto="${msa_dir}/mgnify_hits.sto"
    if [ ! -e ${mgnify_hits_sto} ]; then
      create_sto ${f} ${mgnify_hits_sto}
    fi
    touch ${msa_dir}/pdb_hits.hhr
    #bfd_uniclust_hits_a3m="${msa_dir}/bfd_uniclust_hits.a3m"
    bfd_uniref_hits_a3m="${msa_dir}/bfd_uniref_hits.a3m"
    if [ ! -e ${bfd_uniref_hits_a3m} ]; then
      cp ${f} ${bfd_uniref_hits_a3m}
    fi
    af2_params="--use_precomputed_msas ${af2_params}"
  fi
  python docker/run_docker.py \
      --docker_image_name=${docker_image_name} \
      --data_dir=${data_dir} \
      --max_template_date=${max_template_date} \
      --output_dir=${output_dir} \
      --models model_msa_1 \
      --fasta_fmt=pkl \
      ${pf2_params} \
      ${f}
  echo "Finished (${f}|${msa_mode}) `date`"
done
