<style type="text/css">
body{
    font-family:sans-serif;
}
.value-off {
    background-color: skyblue;
}
.value-on {
    background-color: tomato;
    
}
.value-off:hover{
    background-color: deepskyblue;
    cursor:pointer;
}
.value-on:hover{
    background-color: red;
    cursor:pointer;
}
</style>
<script type="text/javascript" src="js/jquery.js" ></script>
<script type="text/javascript" src="https://www.google.com/jsapi"></script>
<script type="text/javascript">
function toggle(cell){
    if(cell.getAttribute("data-value") == 0){
	setCell(cell);
    } else {
	resetCell(cell);
    }
}

function setCell(cell){
    cell.setAttribute("data-value",1);
    cell.setAttribute("class","value-on");
}

function resetCell(cell){
    cell.setAttribute("data-value",0);
    cell.setAttribute("class","value-off");
}

function resetBoard(){
    $("td[digit-cell=1]").each(function(){resetCell(this);});
}

function load_digit(digit){
    resetBoard();
    $.get('inputs/input'+digit,function(data){
	// alert(data);
	var rows = data.split("\n");
	for(var row_id in rows){
	    var columns = rows[row_id].split(" ");
	    for(var column_id in columns){
		if(parseInt(columns[column_id])){
		    $("td[data-row="+row_id+"][data-column="+column_id+"]").each(function(){
			toggle(this);
		    });
		}
	    }
	}
    });
}


function recognize_digit(){
    var input = [];
    $("td[digit-cell=1]").each(function(){
	input.push(this.getAttribute('data-value'));
    });
    // alert(input);
    $.post('run_input.php',{'input':input}).done(function(data){
	// alert(data);
	var xdata = JSON.parse(data);
	for(var i in xdata){
	    $('td[data-rcon='+i+']').each(function(){
		this.innerHTML = xdata[i].toFixed(2);
	    });
	}
	drawChart(xdata);
    });
}


    google.load("visualization", "1", {packages:["corechart"]});
      // google.setOnLoadCallback(drawChart);
      function drawChart(digit_probabs) {
        var digit_table = [];
        digit_table.push(['digit','probability']);
        for(i in digit_probabs){
    	    digit_table.push([i,digit_probabs[i]]);
        }
        var data = google.visualization.arrayToDataTable(digit_table);

        var options = {
          title: 'Digit probability',
          hAxis: {title: 'Digit', titleTextStyle: {color: 'red'}}
        };

        var chart = new google.visualization.ColumnChart(document.getElementById('chart_div'));
        chart.draw(data, options);
      }



</script>

<?php
echo 'Set input to: <br />';
echo '<div style="width:20px; width:20px; padding:4px; margin:1px; text-align:center; display:inline-block" class="value-on" onclick="resetBoard()">X</div>';
for($k=0;$k<10;$k++){
    echo '<div style="width:20px; width:20px; padding:4px; margin:1px; text-align:center; display:inline-block" class="value-on" onclick="load_digit('.$k.')">'.$k.'</div>';
}
echo '<br /><br />';

echo 'Input:<br />';
echo '<table style="border:0px" border="0">';
for($i=0;$i<8;$i++){
    echo '<tr>';
    for($j=0;$j<8;$j++){
	echo '<td digit-cell=1 data-row="'.$i.'" data-column="'.$j.'" data-value="0" style="width:20px; height:20px" class="value-off" onclick="toggle(this)"></td>';
    }
    echo '</tr>';
}
echo '</table>';

echo '<input type="button" value="Test" onclick="recognize_digit()" />';
echo '<br /><br />';
echo 'Probabilities:<br />';
echo '<table border="0"><tr>';
for($k=0;$k<10;$k++){
    echo'<th style="width:20px">'.$k.'</th>';
}
echo '</tr><tr>';
for($k=0;$k<10;$k++){
    echo'<td data-rcon="'.$k.'">&nbsp;</td>';
}
echo '</tr></table>';

echo '<div id="chart_div"></div>';

include_once 'text.html';
?>