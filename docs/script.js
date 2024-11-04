
const videoPaths = [
    "static/video/demo001.mp4",
    "static/video/demo002.mp4",
    "static/video/demo003.mp4",
    "static/video/demo004.mp4",
    "static/video/demo005.mp4",
]

let currentVideoIndex = 0;

window.onload = function() {
    const videoPlayer = document.querySelector('.video-player');
    const prevBtn = document.getElementById('prev');
    prevBtn.addEventListener('click', function() {
        currentVideoIndex = (currentVideoIndex - 1 + videoPaths.length) % videoPaths.length;
        loadVideo(currentVideoIndex);
    });
    const nextBtn = document.getElementById('next');
    nextBtn.addEventListener('click', function() {
        currentVideoIndex = (currentVideoIndex + 1) % videoPaths.length;
        loadVideo(currentVideoIndex);
    });
    const playPauseBtn = document.querySelector('#play-pause-btn');
    const downloadBtn = document.querySelector('#download-btn');
    loadVideo(currentVideoIndex);
}

function loadVideo(index) {
    const video = document.getElementById('video');
    const videoSource = videoPaths[index];
    video.src = videoSource;
    video.load();
}
