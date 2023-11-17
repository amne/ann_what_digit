<?php
require_once 'ann/c_ann.php';
require_once 'digit_manip.php';
$c_ann = new CAnn(array(),array(),0.3);
$c_ann->read_map_file('map/good_digits_all_rprop');
// echo print_r($_POST,true);
$input = $_POST['input'];
if(count($input)!=64){
    return json_encode(array('F','F','F','F','F','F','F','F','F','F','F'));
}
$output = $c_ann->run(flatten_matrix(trim_digit(input_to_matrix($input,8),8)));
echo json_encode($output);
?>