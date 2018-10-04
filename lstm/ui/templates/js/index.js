$(document).ready(function() {

  let curRow = null;
  let curNeighbour = 0;
  let neighbourTables = {};

  const selectRow = (rowIdx) => {
    if(curRow !== null) {
      $(`tr:eq(${curRow})`).removeClass("highlight-selected");
    }

    curRow = rowIdx;
    $(`tr:eq(${rowIdx})`).addClass("highlight-selected");

  };

  const clearNRows = () => {
    neighbourFnames.forEach(fname => {
      neighbourTables[fname].curRows.forEach(curRow => {
        $($(`#frame_${fname}`).get(0).contentWindow.document).find(`tr:eq(${curRow})`).removeClass("highlight-selected");
      });
    });
  };

  const selectNRow = (fname, rowIdx) => {
    neighbourTables[fname].curRows.push(rowIdx);
    $($(`#frame_${fname}`).get(0).contentWindow.document).find(`tr:eq(${rowIdx})`).addClass("highlight-selected");
  };

  const showNeighbour = (i) => {
    $(`#frame_${neighbourFnames[curNeighbour]}`).css("display", "none");
    curNeighbour = i;
    $(`#frame_${neighbourFnames[curNeighbour]}`).css("display", "block");
  };

  $("tr").click(function() {
    const i = $("tr").index($(this));
    selectRow(i);
    clearNRows();
    const simRows = rowSimMap[i];
    simRows.forEach(rowInfo => {
      selectNRow(rowInfo.fname, rowInfo.row);
    });
    //const el = document.getElementById("neighbour-table").contentWindow;
  });

  $("#prev-table").click(() => {
    let prevNeighbour = curNeighbour - 1;
    if (curNeighbour === 0) {
      prevNeighbour = neighbourFnames.length - 1;
    }
    showNeighbour(prevNeighbour);
  });

  $("#next-table").click(() => {
    let nextNeighbour = curNeighbour + 1;
    if (curNeighbour === neighbourFnames.length - 1) {
      nextNeighbour = 0;
    }
    showNeighbour(nextNeighbour);
  });

  /* ---- Init steps ---- */
  neighbourFnames.forEach(fname => {
    neighbourTables[fname] = {
      curRows: []
    };
    let $iframe = $("<iframe>", {
      id: `frame_${fname}`, 
      width: 800,
      height: 800,
      frameBorder: 0
    });
    const neighbourTableContent = $(`#${fname}`).val(); 
    $iframe.css("display", "none");
    $("#frame-container").append($iframe);
    $iframe.contents().find('html').html(neighbourTableContent);
  });
  //$("#neighbour-table").contents().find('html').html(neighbourTableContent);
  $(`#frame_${neighbourFnames[0]}`).css("display", "block");

});
