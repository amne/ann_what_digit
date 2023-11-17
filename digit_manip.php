<?php

/*
    takes a flat input and turns it into a matrix (ie: 64 inputs, row_len = 8 will output a 8x8 matrix)
*/
function input_to_matrix($input,$row_len = 8){
    $matrix = [];
    $j = 0;
    $k = 1;
    while($j<count($input)){
	$row_num = $j % $row_len;//  floor($j/$row_len);
	$col_num = floor($j / $row_len);
	if(!isset($matrix[$row_num])) $matrix[$row_num] = [];
	$matrix[$row_num][$col_num] = $input[$j];
	$j++;
    }
    return $matrix;
}


/*
    takes a matrix (ie 8x8) and trims the blank space around the digit to normalize it. output is still a 8x8 but digit will be in the top-left corner
*/

function trim_digit($input_matrix,$row_len = 8){
    $x_min = $row_len;
    $y_min = $row_len;
    $x_max = 0;
    $y_max = 0;
    $num_rows = count($input_matrix);
    for($x=0;$x<$row_len;$x++){
	for($y=0;$y<$row_len;$y++){
	    if($input_matrix[$x][$y]){
		if($x<$x_min) $x_min = $x;
		if($y<$y_min) $y_min = $y;
		if($x>$x_max) $x_max = $x;
		if($y>$y_max) $y_max = $y;
	    }
	}
    }
    $result = [];
    for($i=$x_min;$i<=$x_max;$i++){
	$result[$i-$x_min] = [];
	for($j=$y_min;$j<=$y_max;$j++){
	    $result[$i-$x_min][] = $input_matrix[$i][$j];
	}
	for($j=$y_max-$y_min+1;$j<$num_rows;$j++){
	    $result[$i-$x_min][] = 0;
	}
    }
    for($i=$x_max-$x_min+1;$i<$row_len;$i++){
	for($j=0;$j<$num_rows;$j++){
	    $result[$i][$j] = 0;
	}
    }
    
    return $result;
}

/*
    returns a flat input
*/
function flatten_matrix($input_matrix){
    $input = [];
    for($i=0;$i<count($input_matrix);$i++){
	for($j=0;$j<count($input_matrix[$i]);$j++){
	    $input[] = $input_matrix[$i][$j];
	}
    }
    return $input;
}

?>