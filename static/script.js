document.addEventListener("DOMContentLoaded", function () {
    const startBtn = document.getElementById("start-btn");
    const resultBtn = document.getElementById("result-btn");
    const videoFeed = document.getElementById("video-feed");

    startBtn.addEventListener("click", function () {
        videoFeed.src = "/video_feed";
        startBtn.disabled = true;

        // Automatically stop video after 10 seconds and show result button
        setTimeout(function () {
            videoFeed.src = "";
            resultBtn.classList.remove("d-none");
        }, 10000); // 10 seconds
    });
});
