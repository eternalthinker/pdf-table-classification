$(document).ready(function() {

  let curRow = null;

  const selectRow = (rowIdx) => {
    if(curRow !== null) {
      $(`tr:eq(${curRow})`).css("background", "none");
    }

    curRow = rowIdx;
    $(`tr:eq(${rowIdx})`).css("background", "yellow");

  };

  const selectNRow = (rowIdx) => {
    if(curRow !== null) {
      $($(`#neighbour-table`).get(0).contentWindow.document).find(`tr:eq(${curRow})`).css("background", "none");
    }

    curRow = rowIdx;
    $($(`#neighbour-table`).get(0).contentWindow.document).find(`tr:eq(${rowIdx})`).css("background", "yellow");
  };

  $("tr").click(function() {
    const i = $("tr").index($(this));
    selectRow(i);
    selectNRow(i);
    const el = document.getElementById("neighbour-table").contentWindow;
  });

  /* ---- Init steps ---- */
  neighbourFnames.forEach(fname => {
    let $iframe = $("<iframe>", {
      id: `frame_${fname}`, 
      width: 1000,
      height: 400
    });
    const neighbourTableContent = $(`#${fname}`).val(); 
    $iframe.css("display", "none");
    $("#frame-container").append($iframe);
    $iframe.contents().find('html').html(neighbourTableContent);
  });
  //$("#neighbour-table").contents().find('html').html(neighbourTableContent);
  $(`#frame_${neighbourFnames[0]}`).css("display", "block");

});

