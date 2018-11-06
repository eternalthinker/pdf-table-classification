$(document).ready(function() {

  let curRow = null;
  let curNeighbour = 0;
  let neighbourTables = {};

  const selectRow = (rowIdx, className) => {
    if(curRow !== null) {
      $(`tr:eq(${curRow})`).removeClass(className);
    }

    curRow = rowIdx;
    $(`tr:eq(${rowIdx})`).addClass(className);
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
    if (dist < 10) {
      $selectedRow.addClass("highlight-selected");
    } else if (dist < 1000) {
      $selectedRow.addClass("highlight2-selected");
    }
    $selectedRow.attr('title', `${dist}`);
  };

  const clearQueryResults = () => {
    neighbourFnames.forEach(fname => {
      neighbourTables[fname].queryResults.forEach(curRow => {
        $($(`#frame_${fname}`).get(0).contentWindow.document).find(`tr:eq(${curRow})`)
          .removeClass("query-result");
      });
    });
  };

  const selectQueryResult = (fname, rowIdx) => {
    neighbourTables[fname].queryResults.push(rowIdx);
    let $selectedRow = $($(`#frame_${fname}`).get(0).contentWindow.document).find(`tr:eq(${rowIdx})`);
    $selectedRow.addClass("query-result");
  };

  const showNeighbour = (i) => {
    $(`#frame_${neighbourFnames[curNeighbour]}`).css("display", "none");
    curNeighbour = i;
    $(`#frame_${neighbourFnames[curNeighbour]}`).css("display", "block");
    $('#resultTableId').html(`${curNeighbour + 1}`);
  };

  $("tr").click(function() {
    const i = $("tr").index($(this));
    selectRow(i, 'highlight-selected');
    clearNRows();
    const simRows = rowSimMap[i];
    simRows.forEach(rowInfo => {
      selectNRow(rowInfo.fname, rowInfo.row, rowInfo.dist);
    });
    //const el = document.getElementById("neighbour-table").contentWindow;
  });

  function callApi({endpoint, method, data}) {
    let params = {};
    if (method === 'POST') {
      params = {
        method: 'POST',
        body: JSON.stringify(data),
        headers: {
          'Content-Type': 'application/json'
        }
      };
    }
    return fetch(endpoint, params)
      .then(res => res.json())
      .then(json => json);
  }

  $("#queryStringSubmit").click((event) => {
    event.preventDefault();
    event.stopPropagation();
    const queryString = $('#queryString').val();
    callApi({
      endpoint: `/query`,
      method: 'POST',
      data: {
        query: queryString
      }
    }).then(resultTablesMap => {
      clearQueryResults();
      neighbourFnames.forEach(fname => {
        resultTablesMap[fname].forEach(curRow => {
          selectQueryResult(fname, curRow);
        });
      });
    }); 
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
      curRows: [],
      queryResults: []
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
  $('#resultTableId').html('1');
  $('.loading').css('visibility', 'hidden');
});

