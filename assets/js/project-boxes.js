document.querySelectorAll('.toggle-btn').forEach(button => {
    button.addEventListener('click', () => {
        const projectBox = button.closest('.project-box');
        const description = projectBox.querySelector('.project-description');
        const isExpanded = description.classList.contains('show');

        // Close all other open descriptions
        document.querySelectorAll('.project-box').forEach(box => {
            if (box !== projectBox) {
                box.classList.remove('expanded');
                box.querySelector('.project-description').classList.remove('show');
                box.querySelector('.toggle-btn').textContent = 'Show Description';
            }
        });

        // Toggle the clicked project box
        projectBox.classList.toggle('expanded', !isExpanded);
        description.classList.toggle('show', !isExpanded);
        button.textContent = isExpanded ? 'Show Description' : 'Hide Description';

        // Scroll to the expanded box if it's not in view
        if (!isExpanded) {
            projectBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    });
});