<script>
window.onload = function() {
  // 查找所有 div.table-container 元素
  const tableContainers = document.querySelectorAll(".table-container");
  // 遍历所有 div.table-container 元素
  tableContainers.forEach(function(tableContainer) {
    // 将 div.table-container 的类名设置为 "code-hidden"
    tableContainer.classList.add("code-hidden");

    // 获取 div.table-container 内的 span 元素数量
    const spanCount = tableContainer.querySelectorAll("tbody > tr > td.code > pre > span").length;

    // 检查 span 元素数量是否 >= 1
    if (spanCount >= 5) {
      // 创建一个 <span> 标签并添加功能
      const spanElement = document.createElement("span");
      spanElement.innerHTML = "The code is below 👇"; // 表示折叠代码

      // 插入一点空格
      const spaceElement = document.createTextNode(" ");
      spanElement.insertBefore(spaceElement, spanElement.firstChild);

      // 将 span 插入到 DOM 中
      tableContainer.parentNode.insertBefore(spanElement, tableContainer);

      // 为 span 标签添加点击事件
      spanElement.addEventListener("click", function() {
        // 切换 tableContainer 的 "code-hidden" 类
        tableContainer.classList.toggle("code-hidden");

        // 切换 span 标签的内容
        spanElement.innerHTML = tableContainer.classList.contains("code-hidden") ? "The code is below 👇" : "Hide the code ☝";
      });
    }
  });
};
</script>

