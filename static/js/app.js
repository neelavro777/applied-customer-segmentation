// Preset chip -> fill the form
document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
        document.getElementById('recency').value = chip.dataset.r;
        document.getElementById('frequency').value = chip.dataset.f;
        document.getElementById('monetary').value = chip.dataset.m;
    });
});

// Submit button loading state
const form = document.getElementById('predict-form');
if (form) {
    form.addEventListener('submit', () => {
        const btn = document.getElementById('submit-btn');
        btn.disabled = true;
        btn.textContent = 'Thinking...';
    });
}

// Charts (only when a result is present)
const dataEl = document.getElementById('result-data');
if (dataEl && window.Chart) {
    const data = JSON.parse(dataEl.textContent);

    const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim() || '#3b82f6';
    const grid = 'rgba(148,163,184,0.15)';
    const tick = '#94a3b8';

    // Radar: input vs typical customer in the segment
    new Chart(document.getElementById('radarChart'), {
        type: 'radar',
        data: {
            labels: ['Recency', 'Frequency', 'Monetary'],
            datasets: [
                {
                    label: 'This customer',
                    data: data.norm_input,
                    borderColor: cssVar('--primary'),
                    backgroundColor: 'rgba(59,130,246,0.25)',
                    pointBackgroundColor: cssVar('--primary'),
                },
                {
                    label: 'Typical for this group',
                    data: data.norm_centroid,
                    borderColor: cssVar('--primary-2'),
                    backgroundColor: 'rgba(139,92,246,0.15)',
                    pointBackgroundColor: cssVar('--primary-2'),
                },
            ],
        },
        options: {
            responsive: true,
            plugins: { legend: { labels: { color: tick } } },
            scales: {
                r: {
                    min: 0, max: 1,
                    angleLines: { color: grid },
                    grid: { color: grid },
                    pointLabels: { color: tick, font: { size: 12 } },
                    ticks: { display: false, backdropColor: 'transparent' },
                },
            },
        },
    });

    // Distance bars: how close to every group
    const labels = data.per_cluster.map(c => c.name);
    const values = data.per_cluster.map(c => c.distance);
    const colors = data.per_cluster.map(c => c.is_assigned ? cssVar('--primary') : 'rgba(148,163,184,0.35)');

    new Chart(document.getElementById('distChart'), {
        type: 'bar',
        data: { labels, datasets: [{ label: 'Distance', data: values, backgroundColor: colors, borderRadius: 6 }] },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: grid }, ticks: { color: tick } },
                y: { grid: { display: false }, ticks: { color: tick } },
            },
        },
    });
}
