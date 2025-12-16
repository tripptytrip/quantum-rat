
document.addEventListener('DOMContentLoaded', () => {
    const runSelector = document.getElementById('run-selector');
    const episodeSelector = document.getElementById('episode-selector');
    const summaryJson = document.getElementById('summary-json');
    const tickScrubber = document.getElementById('tick-scrubber');
    const tickDisplay = document.getElementById('tick-display');
    const tickJson = document.getElementById('tick-json');

    let currentRun = null;
    let currentEpisode = null;
    let tickCache = [];
    let totalTicks = 0;

    // Fetch and populate runs
    fetch('/api/runs')
        .then(response => response.json())
        .then(data => {
            runSelector.innerHTML = '<option value="">Select a run</option>';
            data.runs.forEach(run => {
                const option = document.createElement('option');
                option.value = run.id;
                option.textContent = `${run.id} (${run.type})`;
                runSelector.appendChild(option);
            });
        });

    runSelector.addEventListener('change', () => {
        currentRun = runSelector.value;
        if (!currentRun) {
            episodeSelector.innerHTML = '';
            return;
        }
        fetch(`/api/runs/${currentRun}/episodes`)
            .then(response => response.json())
            .then(data => {
                episodeSelector.innerHTML = '<option value="">Select an episode</option>';
                data.episodes.forEach(ep => {
                    const option = document.createElement('option');
                    option.value = ep.episode_id;
                    option.textContent = ep.episode_id;
                    option.dataset.agentId = ep.agent_id;
                    option.dataset.protocol = ep.protocol;
                    episodeSelector.appendChild(option);
                });
            });
    });

    episodeSelector.addEventListener('change', () => {
        const selectedOption = episodeSelector.options[episodeSelector.selectedIndex];
        if (!selectedOption.value) return;

        currentEpisode = {
            id: selectedOption.value,
            agent_id: selectedOption.dataset.agentId,
            protocol: selectedOption.dataset.protocol,
        };
        
        loadEpisode(currentRun, currentEpisode);
    });

    tickScrubber.addEventListener('input', () => {
        const tickIndex = parseInt(tickScrubber.value, 10);
        tickDisplay.textContent = tickIndex;
        displayTick(tickIndex);
    });

    function loadEpisode(runId, episode) {
        // Fetch summary
        fetch(`/api/runs/${runId}/${episode.agent_id}/${episode.protocol}/summary`)
            .then(response => response.json())
            .then(summary => {
                summaryJson.textContent = JSON.stringify(summary, null, 2);
            });

        // Fetch ticks and setup UI
        fetchTicks(runId, episode.agent_id, episode.protocol, 0, 10000).then(() => { // Fetch all ticks for now
             updatePlots();
        });
    }
    
    async function fetchTicks(runId, agentId, protocol, start = 0, limit = 500) {
        const response = await fetch(`/api/runs/${runId}/${agentId}/${protocol}/ticks?start=${start}&limit=${limit}`);
        const data = await response.json();
        tickCache = data.ticks;
        totalTicks = data.total;
        tickScrubber.max = totalTicks - 1;
        tickScrubber.value = 0;
        tickDisplay.textContent = 0;
        displayTick(0);
        return data;
    }

    function displayTick(index) {
        if (tickCache[index]) {
            tickJson.textContent = JSON.stringify(tickCache[index], null, 2);
        } else {
            // In a real app, you would fetch the page containing the tick
            tickJson.textContent = "Tick not loaded in cache.";
        }
    }
    
    function updatePlots() {
        const kappa = tickCache.map(t => t.kappa);
        const avalanche_size = tickCache.map(t => t.avalanche_size);
        const microsleep_active = tickCache.map(t => t.microsleep_active ? 1 : 0);
        const replay_active = tickCache.map(t => t.replay_active ? 1 : 0);
        const grid_x = tickCache.map(t => t.grid_x);
        const grid_y = tickCache.map(t => t.grid_y);
        const ticks = tickCache.map((t, i) => i);

        Plotly.newPlot('kappa-plot', [{ x: ticks, y: kappa, mode: 'lines', name: 'Kappa' }], { title: 'Kappa over Time' });
        Plotly.newPlot('avalanche-plot', [{ x: ticks, y: avalanche_size, type: 'scatter', mode: 'markers', name: 'Avalanche Size' }], { title: 'Avalanche Size over Time' });
        Plotly.newPlot('microsleep-plot', [
            { x: ticks, y: microsleep_active, type: 'scatter', mode: 'lines', name: 'Microsleep' },
            { x: ticks, y: replay_active.map(v => v * 0.9), type: 'scatter', mode: 'lines', name: 'Replay' }
        ], { title: 'Microsleep/Replay Timeline' });
        Plotly.newPlot('spatial-plot', [{ x: grid_x, y: grid_y, mode: 'lines', name: 'Path' }], { title: 'Spatial Trace' });

        const actionNames = tickCache.map(t => t.action_name);
        const actionCounts = actionNames.reduce((acc, name) => {
            acc[name] = (acc[name] || 0) + 1;
            return acc;
        }, {});
        Plotly.newPlot('action-plot', [{ x: Object.keys(actionCounts), y: Object.values(actionCounts), type: 'bar' }], { title: 'Action Frequency' });
    }
});
