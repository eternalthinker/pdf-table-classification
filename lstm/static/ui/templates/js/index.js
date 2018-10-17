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
        $($(`#frame_${fname}`).get(0).contentWindow.document).find(`tr:eq(${curRow})`)
          .removeAttr('title')
          .removeClass("highlight-selected")
          .removeClass("highlight2-selected");
      });
    });
  };

  const selectNRow = (fname, rowIdx, dist) => {
    neighbourTables[fname].curRows.push(rowIdx);
    let $selectedRow = $($(`#frame_${fname}`).get(0).contentWindow.document).find(`tr:eq(${rowIdx})`);
    if (dist < 100) {
      $selectedRow.addClass("highlight-selected");
    } else if (dist < 200) {
      $selectedRow.addClass("highlight2-selected");
    }
    $selectedRow.attr('title', `${dist}`);
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
      selectNRow(rowInfo.fname, rowInfo.row, rowInfo.dist);
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

  $('#queryTableIdx').on('change', function() {
    $('.loading').css('visibility', 'visible');
    $(this).closest('form').submit();
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
  $('.loading').css('visibility', 'hidden');
});

