
function inputTipText() {
    $("input[class*=grayTips]") //所有样式名中含有grayTips的input  
        .each(function() {  
            var oldVal=$(this).val();   //默认的提示性文本  
            $(this)  
                .css({"color":"#888"})  //灰色  
                .focus(function() {  
                    if ($(this).val() != oldVal) {
                        $(this).css({"color":"#000"});
                    } else {
                        $(this).val("").css({"color":"#888"});
                    }
                })
                .blur(function() {
                    if ($(this).val() == "") {
                        $(this).val(oldVal).css({"color":"#888"});
                    }
                })
                .keydown(function() {
                    $(this).css({"color":"#000"});
                })
        });
}

$(function() {
    inputTipText();  //直接调用就OK了  
});


function wangbing_clear(){ 
    document.clearrecentjobs.action="/TreeThreaderV2/clear.php"
    document.clearrecentjobs.submit();
}

function wangbing_remove(){
    document.clearrecentjobs.action="/TreeThreaderV2/remove.php"
    document.clearrecentjobs.submit();
}

function wangbing_jobid(jobid){
        var id = document.getElementById("job_input").value;
        if (id == "")
        {
            document.getElementById("job_input").value = jobid;
        }
}


function deselectjobs(num)
{
    var number = $(clear_checkboxes).value;
    if (typeof num ==='undefined' )
    {
        number = 1;
    }else{
        number = num;   //wangbing
    }
    if (number==1) {
    document.clearrecentjobs.elements["jobid[]"].checked = false;
    } else {
    for(i=0; i<number; i++) {
        document.clearrecentjobs.elements["jobid[]"][i].checked = false;
    }
    }
}

function selectjobs(num)
{ 
    var number = $(clear_checkboxes).value;
    if (typeof num ==='undefined' )
    {
        number = 1;
    }else{
        number = num;   //wangbing
    }
    if (number==1) {
    document.clearrecentjobs.elements["jobid[]"].checked = true;
    } else {
    for(i=0; i<number; i++) {
        document.clearrecentjobs.elements["jobid[]"][i].checked = true;
    }
    }
}
