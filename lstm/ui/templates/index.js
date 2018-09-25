$(document).ready(function() {

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
    selectRow(i);
    const el = document.getElementById("neighbour-table").contentWindow;
    console.log(el);
  });

});

