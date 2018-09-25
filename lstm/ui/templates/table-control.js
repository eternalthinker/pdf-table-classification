/*let curRow = null;

function selectRow (rowIdx) {
  if(curRow !== null) {
    $(`tr:eq(${curRow})`).css("background", "none");
  }

  curRow = rowIdx;
  $(`tr:eq(${rowIdx})`).css("background", "yellow");
}

selectRow(3);*/

$(document).ready(function() {
    console.log("dsdsd")

  let curRow = null;

  const selectRow = (rowIdx) => {
    if(curRow !== null) {
      $(`tr:eq(${curRow})`).css("background", "none");
    }

    curRow = rowIdx;
    $(`tr:eq(${rowIdx})`).css("background", "yellow");
  };

  const neighbourTableContent = $("textarea:eq(1)").val();  
  $("#neighbour-table").contents().find('html').html(neighbourTableContent);

  $("tr").click(function() {
    const i = $("tr").index($(this));
    console.log(i);
    selectRow(i);
  });

});

