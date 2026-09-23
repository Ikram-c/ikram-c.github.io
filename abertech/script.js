/* 
   AberTech — panel navigation & project previews
   Project links open their GitHub repositories.
   Touch devices: first tap previews a project, second tap opens the repo.
*/

// ── Navigation: Panel Swapping ──
const navLinks = document.querySelectorAll('[data-nav]');
const panels = document.querySelectorAll('.panel');
const canvasImages = document.querySelectorAll('.canvas-image');
const canvasContexts = document.querySelectorAll('.canvas-context');
const placeholder = document.getElementById('placeholder');
let currentPanel = 'home';
let currentSlide = -1;

// Initial state: Home panel is active in the HTML.
placeholder.classList.add('hidden');
const homeContext = document.querySelector('[data-context="home"]');
if (homeContext) {
   homeContext.classList.add('visible');
}

function switchPanel(target) {
   if (target === currentPanel) return;
   currentPanel = target;
   currentSlide = -1;

   // Update nav active state
   navLinks.forEach(link => {
      link.classList.toggle('nav-active', link.dataset.nav === target);
   });

   // Switch left panel
   panels.forEach(panel => {
      panel.classList.toggle('panel-active', panel.id === 'panel-' + target);
   });

   // Reset canvas
   canvasImages.forEach(img => img.classList.remove('visible'));
   canvasContexts.forEach(ctx => ctx.classList.remove('visible'));

   if (target === 'work') {
      placeholder.classList.remove('hidden');
   } else {
      placeholder.classList.add('hidden');
      const contextImg = document.querySelector('[data-context="' + target + '"]');
      if (contextImg) {
         setTimeout(() => contextImg.classList.add('visible'), 80);
      }
   }
}

navLinks.forEach(link => {
   link.addEventListener('click', (e) => {
      e.preventDefault();
      switchPanel(link.dataset.nav);
      if (window.scrollY > 10) {
         window.scrollTo({ top: 0, behavior: 'smooth' });
      }
   });
});

// ── Projects: Hover Preview ──
const projectItems = document.querySelectorAll('.project-item');
const slides = document.querySelectorAll('.canvas-image');

projectItems.forEach(item => {
   item.addEventListener('mouseenter', () => {
      if (currentPanel !== 'work') return;
      const idx = parseInt(item.dataset.index, 10);
      if (idx === currentSlide) return;
      currentSlide = idx;

      placeholder.classList.add('hidden');

      slides.forEach(slide => {
         slide.classList.toggle('visible', parseInt(slide.dataset.slide, 10) === idx);
      });
   });
});

document.querySelector('#panel-work').addEventListener('mouseleave', () => {
   if (currentPanel !== 'work') return;
   currentSlide = -1;
   slides.forEach(s => s.classList.remove('visible'));
   placeholder.classList.remove('hidden');
});

// ── Mobile: Touch support ──
if (window.matchMedia('(hover: none)').matches) {
   projectItems.forEach(item => {
      item.addEventListener('click', (e) => {
         if (currentPanel !== 'work') return;
         if (item.classList.contains('active')) {
            return; // allow default navigation to repo
         }
         e.preventDefault();
         const idx = parseInt(item.dataset.index, 10);

         placeholder.classList.add('hidden');
         projectItems.forEach(i => i.classList.remove('active'));
         item.classList.add('active');

         slides.forEach(slide => {
            const match = parseInt(slide.dataset.slide, 10) === idx;
            slide.classList.toggle('visible', match);
            if (match && window.innerWidth <= 900) {
               slide.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
         });
      });
   });
}

// ── Theme Switcher Logic ──
(function () {
    var root    = document.documentElement;
    var buttons = document.querySelectorAll('.theme-switch button[data-theme-set]');

    function current() {
        var v = root.getAttribute('data-theme');
        return (v === 'light' || v === 'dark') ? v : 'auto';
    }

    function apply(state) {
        if (state === 'auto') {
            root.removeAttribute('data-theme');
            localStorage.removeItem('theme');
        } else {
            root.setAttribute('data-theme', state);
            localStorage.setItem('theme', state);
        }
        buttons.forEach(function (btn) {
            var active = btn.dataset.themeSet === state;
            btn.classList.toggle('is-active', active);
            btn.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
    }

    apply(current());

    buttons.forEach(function (btn) {
        btn.addEventListener('click', function () {
            apply(btn.dataset.themeSet);
        });
    });
})();
