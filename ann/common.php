<?php
require_once dirname(__FILE__).'/damn_debug.php';
    function doublespace($string){
	while(strpos($string,'  ')!==false) $string = strtr($string,array('  '=>' '));
	return $string;
    }
    
    
    function activation($x){
	return 1/(1+pow(M_E,-$x));
    }
    
    function activation_D($x){
	return activation($x)*(1-activation($x));
    }
    
    
    function random_order($max=4){
    $order = array();
    $k = 0;
    do{
    $r = rand(0,$max-1);
    if(array_search($r,$order)===false){
	$order[$k] = $r;
	$k++;
    }
    }while($k<$max);
    return $order;
    }

    
    
?>