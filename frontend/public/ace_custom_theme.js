ace.define("ace/theme/custom_diff", ["require","exports","module","ace/lib/dom"], function(require, exports, module) {
  exports.isDark = true;
  exports.cssClass = "ace-custom-diff";
  exports.cssText = `
    .ace-custom-diff .ace_gutter {background: #2f3129; color: #8f908a;}
    .ace-custom-diff .ace_print-margin {width: 1px; background: #555651;}
    .ace-custom-diff {background-color: #272822; color: #f8f8f2;}
    .ace-custom-diff .ace_cursor {color: #f8f8f0;}
    .ace-custom-diff .ace_marker-layer .added-line {
        position: absolute;
        z-index: 20;
        background: rgba(46,204,113,0.25);
    }
    .ace-custom-diff .ace_marker-layer .deleted-line {
        position: absolute;
        z-index: 20;
        background: rgba(231,76,60,0.25);
    }
    .ace-custom-diff .ace_marker-layer .modified-line {
        position: absolute;
        z-index: 20;
        background: rgba(241,196,15,0.25);
    }
  `;
  var dom = require("../lib/dom");
  dom.importCssString(exports.cssText, exports.cssClass);
});
