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

  const neighbourTableContent = $("textarea:eq(1)").val();  
  $("#neighbour-table").contents().find('html').html(neighbourTableContent);

  $("tr").click(function() {
    const i = $("tr").index($(this));
    selectRow(i);
    selectNRow(i);
    const el = document.getElementById("neighbour-table").contentWindow;
  });

});

