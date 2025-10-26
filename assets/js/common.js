$(document).ready(function() {
  // add toggle functionality to abstract and bibtex buttons
  $('a.abstract').click(function() {
    $(this).parent().parent().find(".abstract.hidden").toggleClass('open');
    $(this).parent().parent().find(".bibtex.hidden.open").toggleClass('open');
  });
  $('a.bibtex').click(function() {
    $(this).parent().parent().find(".bibtex.hidden").toggleClass('open');
    $(this).parent().parent().find(".abstract.hidden.open").toggleClass('open');
  });
  $('a').removeClass('waves-effect waves-light');

  // bootstrap-toc
  if($('#toc-sidebar').length){
    var navSelector = "#toc-sidebar";
    var $myNav = $(navSelector);
    Toc.init($myNav);
    $("body").scrollspy({
      target: navSelector,
    });
  }

  // add css to jupyter notebooks
  const cssLink = document.createElement("link");
  cssLink.href  = "../css/jupyter.css";
  cssLink.rel   = "stylesheet";
  cssLink.type  = "text/css";

  let theme = localStorage.getItem("theme");
  if (theme == null || theme == "null") {
    const userPref = window.matchMedia;
    if (userPref && userPref("(prefers-color-scheme: dark)").matches) {
      theme = "dark";
    }
  }

  $('.jupyter-notebook-iframe-container iframe').each(function() {
    $(this).contents().find("head").append(cssLink);

    if (theme == "dark") {
      $(this).bind("load",function(){
        $(this).contents().find("body").attr({
          "data-jp-theme-light": "false",
          "data-jp-theme-name": "JupyterLab Dark"});
      });
    }
  });

  const navbar = document.getElementById('navbar');
  if (navbar) {
    let isAnimating = false;

    const handleNavbarScroll = () => {
      if (window.scrollY > 80) {
        if (!navbar.classList.contains('navbar-scrolled') && !isAnimating) {
          navbar.classList.add('navbar-animate');
          navbar.getBoundingClientRect(); // force reflow before expanding
          navbar.classList.add('navbar-scrolled');
          isAnimating = true;
        }
      } else {
        navbar.classList.remove('navbar-scrolled');
        navbar.classList.remove('navbar-animate');
        isAnimating = false;
      }
    };

    navbar.addEventListener('transitionend', (event) => {
      if (event.propertyName === 'width' && navbar.classList.contains('navbar-scrolled')) {
        navbar.classList.remove('navbar-animate');
        isAnimating = false;
      }
    });

    handleNavbarScroll();
    window.addEventListener('scroll', handleNavbarScroll, { passive: true });
  }
});
