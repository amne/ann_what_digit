<?php
define('_PRECISION',6);
class CNeu {
    var $threshold = 0;
    var $base = 0;
    var $output = 0;
    var $strength;
    function __construct($base = 0,$threshold = 0.2){
	$this->base = $base;
	$this->threshold = $threshold;
    }
    function poke($strength){
	$this->strength = $strength; // for training purposes
	$y = activation($strength);
	damn(10,'poked with '.$strength.' on base '.$this->base.' => '.$y.' (|'.$this->threshold.')');
	$this->output = round($y,_PRECISION);
    }
}
?>