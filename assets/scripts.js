document.addEventListener('DOMContentLoaded', function () {
    console.log('DOM fully loaded and parsed');

    const screenWidthRef = 1280; // pixels
    const currentScreenWidth = window.innerWidth; // pixels
    const globalZoomFactor = currentScreenWidth / screenWidthRef;
    document.documentElement.style.setProperty('--global-zoom-factor', globalZoomFactor)

    function addNavLinkListeners() {
        const navLinks = document.querySelectorAll('nav .nav-link');

        navLinks.forEach(link => {
            link.addEventListener('click', function(event) {
                event.preventDefault();

                const targetId = link.getAttribute('href').substring(1);
                const targetSection = document.getElementById(targetId);

                if (targetSection) {
                    window.scrollTo({
                        top: targetSection.offsetTop,
                        behavior: 'smooth'
                    });

                    // Remove active class from all links and add it to the clicked link
                    navLinks.forEach(navLink => navLink.classList.remove('active'));
                    link.classList.add('active');
                }
            });
        });
    }

    function updateNavBarOnScroll() {
        let sections = document.querySelectorAll('section');
        let navLinks = document.querySelectorAll('nav .nav-link');
        let top = window.scrollY;

        sections.forEach(sec => {
            let offset = sec.offsetTop - 50; // Adjusting for navbar height
            let height = sec.offsetHeight;
            let id = sec.getAttribute('id');

            if (top >= offset && top < offset + height) {
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    let activeLink = document.querySelector('nav .nav-link[href*=' + id + ']');
                    if (activeLink) {
                        activeLink.classList.add('active');
                    }
                });
            }
        });
    }

    const observer = new MutationObserver(function (mutations) {
        mutations.forEach(function (mutation) {
            if (mutation.addedNodes.length > 0) {
                addNavLinkListeners();
            }
        });
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });

    addNavLinkListeners();

    window.addEventListener('scroll', updateNavBarOnScroll);

    let firstNavLink = document.querySelector('nav .nav-link');
    if (firstNavLink) {
        firstNavLink.classList.add('active');
    }

    updateNavBarOnScroll();

    setTimeout(function() {
        document.getElementById('loading-mask').style.display = 'none';
    }, 3000); // 5000 ms = 5 secondes
});
