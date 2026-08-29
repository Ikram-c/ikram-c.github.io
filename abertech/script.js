/* AberTech — panel navigation & project previews
   Adapted from TemplateMo 616 Split Index.
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

// Initial state: Home panel is active in the HTML, so show the home
// context card and hide the "Select a project" placeholder.
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
      // If the page is scrolled, bring the split hero back into view.
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
// First tap on a project previews it; a second tap on the same
// (already active) project follows the link to its repository.
// Only for devices WITHOUT a hover pointer (true touch devices) —
// touchscreen laptops with a mouse keep normal click-to-open behaviour.
if (window.matchMedia('(hover: none)').matches) {
   projectItems.forEach(item => {
      item.addEventListener('click', (e) => {
         if (currentPanel !== 'work') return;
         if (item.classList.contains('active')) {
            return; // second tap: allow default navigation to the repo
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
               slide.scrollIntoView({
                  behavior: 'smooth',
                  block: 'nearest'
               });
            }
         });
      });
   });
}
