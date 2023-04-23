{%- if theme.pdf.enable %}
{%- set pdfobject_uri = theme.vendors.pdfobject or '//cdn.jsdelivr.net/npm/pdfobject@2/pdfobject.min.js' %}
<script{{ pjax }}>
if (document.querySelectorAll('div.pdf').length) {
  NexT.utils.getScript('{{ pdfobject_uri }}', () => {
    document.querySelectorAll('div.pdf').forEach(element => {
      embed = function(url, targetSelector, options) {
              if (typeof url !== "string") {
                  return embedError("URL is not valid");
              }
              targetSelector = (typeof targetSelector !== "undefined") ? targetSelector : false;
              options = (typeof options !== "undefined") ? options : {};
              var id = (options.id && typeof options.id === "string") ? "id='" + options.id + "'" : ""
                , page = (options.page) ? options.page : false
                , pdfOpenParams = (options.pdfOpenParams) ? options.pdfOpenParams : {}
                , fallbackLink = (typeof options.fallbackLink !== "undefined") ? options.fallbackLink : true
                , width = (options.width) ? options.width : "100%"
                , height = (options.height) ? options.height : "100%"
                , assumptionMode = (typeof options.assumptionMode === "boolean") ? options.assumptionMode : true
                , forcePDFJS = (typeof options.forcePDFJS === "boolean") ? options.forcePDFJS : false
                , PDFJS_URL = (options.PDFJS_URL) ? options.PDFJS_URL : false
                , targetNode = getTargetElement(targetSelector)
                , fallbackHTML = ""
                , pdfOpenFragment = ""
                , fallbackHTML_default = "<p>This browser does not support inline PDFs. Please download the PDF to view it: <a href='[url]'>Download PDF</a></p>";
              if (!targetNode) {
                  return embedError("Target element cannot be determined");
              }
              if (page) {
                  pdfOpenParams.page = page;
              }
              pdfOpenFragment = buildFragmentString(pdfOpenParams);
              if (forcePDFJS && PDFJS_URL) {
                  return generatePDFJSiframe(targetNode, url, pdfOpenFragment, PDFJS_URL, id);
              } else if (supportsPDFs || (assumptionMode && isModernBrowser && !isIOS)) {
                  return generateEmbedElement(targetNode, targetSelector, url, pdfOpenFragment, width, height, id);
              } else if (PDFJS_URL) {
                  return generatePDFJSiframe(targetNode, url, pdfOpenFragment, PDFJS_URL, id);
              } else {
                  if (fallbackLink) {
                      fallbackHTML = (typeof fallbackLink === "string") ? fallbackLink : fallbackHTML_default;
                      targetNode.innerHTML = fallbackHTML.replace(/\[url\]/g, url);
                  }
                  return embedError("This browser does not support embedded PDFs");
              }
          };
    });
  }, window.PDFObject);
}
</script>
{%- endif %}
