(function() {
    //不同的日期显示不同的样式，200 天为黄色提示，400天为红色提示，可以自己定义。
    let warningDay = 30;
    let errorDay = 365;
    // 确保能够获取到文章时间以及在文章详情页
    let times = document.getElementsByTagName('time');
    if (times.length === 0) { return; }
    let posts = document.getElementsByClassName('post-body');
    if (posts.length === 0) { return; }

    // 获取系统当前的时间
    let pubTime = new Date(times[0].dateTime); /* 文章发布时间戳 */
    let now = Date.now() /* 当前时间戳 */
    let interval = parseInt(now - pubTime)
    let days = parseInt(interval / 86400000)
        /* 发布时间超过指定时间（毫秒） */
        //note warning 以及 note danger 是 Next 主题的自定义模板语法，如果使用其他主题，请自行更改样式以达到最佳显示效果
    if (interval > warningDay * 3600 * 24 * 1000 && interval < errorDay * 3600 * 24 * 1000) {
        posts[0].innerHTML = '<div class="note warning">' +
          '<h5>Reminder: Article Age</h5><p>This article was published ' + days + ' days ago, and some information may have changed. Please keep this in mind when reading.</p>' +
            '</div>' + posts[0].innerHTML;
    } else if (interval >= errorDay * 3600 * 24 * 1000) {
        posts[0].innerHTML = '<div class="note danger">' +
          '<h5>Reminder: Article may be outdated</h5><p>This article was published ' + days + ' days ago, and some information may have changed. Please use discretion when referencing it.</p>' +
            '</div>' + posts[0].innerHTML;
    }
})();
