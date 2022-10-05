// preloader
$(window).on('load',function(){
    $('.preloader').fadeOut(1000); // set duration in brackets
});

$('#Server_Tab').on('click',function(){
  location.href= $(this).attr('href');
});


$(function() {
    new WOW().init();
    $('.templatemo-nav').singlePageNav({
    	offset: 70
    });

    /* Hide mobile menu after clicking on a link
    -----------------------------------------------*/
    $('.navbar-collapse a').click(function(){
        //$(".navbar-collapse").collapse('hide');
    });
})

function addFavorite(siteUrl, siteName) {
    try {
        if (window.external && "addFavorite" in window.external) { // IE
            window.external.addFavorite(siteUrl, siteName);
        } else if (window.sidebar) { // Firefox
            window.sidebar.addPanel(siteName, siteUrl,'');
        } else { // Opera, Chrome, Safari
            alert("Pressing Ctrl + D creates a new bookmark or favorite for the current page");
        }
    } catch (e) {
        alert("Pressing Ctrl + D creates a new bookmark or favorite for the current page");
    }
}
