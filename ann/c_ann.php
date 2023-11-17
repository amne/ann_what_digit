<?php
require_once dirname(__FILE__).'/common.php';
require_once dirname(__FILE__).'/c_neu.php';

class CAnn {
    /*
	map = (layer_0_neuron_count, layer_1_neuron_count, ...)
	weights = array(layer_1=>array(neu_0 => array(weight_from_layer_0_neu_0,weight_from_layer_0_neu1,...),neu_1=>array(weight_from_layer_0_neu_0,...),...),layer_2=>array(...),...)
    */
    var $map;
    var $terrain;
    var $weights = array();
    var $delta_weights = array();
    var $last_weight_change = array();
    var $last_bias_change = array();
    var $last_gradients = array();
    var $bias_weights = array();
    var $bias_delta_weights = array();
    var $epsilon = 0.8; // learning rate
    var $alpha = 0.2; // momentum
    var $workspace;
    var $precision = 0.2;
    var $error_stupid_threshold = 0.5; // if error bigger on an output node bounce the delta by a factor of 10 to 1000 (rand)
    var $last_output = array();
    var $eta_positive = 1.2;
    var $eta_negative = 0.5;
    var $max_delta_step = 1;
    var $min_delta_step = 0.000001;
    var $initial_update = 0.1;
    
    
    
    function __construct($map = array(1),$weights = array(),$precision = 0.2){ // simples ann
	$this->map = $map;
	$this->weights = $weights;
	$this->precision = $precision;
	$this->buildTerrain();
    }
    
    function read_map_file($map_name = 'map'){
	$map_data = file(dirname(__FILE__).'/'.$map_name);
	$map = explode(' ',trim(doublespace($map_data[0])));
	damn(5,'read map data: '.implode(',',$map));
	$weights = array();
	$last_gradients = array();
	foreach($map as $row => $layer_neu_count){
	    if(!$row) continue;
	    $data = $map_data[$row];
	    if($row == 0) continue; // skip the first line. it's map data.
	    $layer_weights = explode(' ',trim(doublespace($map_data[$row])));
	    damn(10,'read layer '.$row.' weights: '.implode(',',$layer_weights));
	    if(!isset($map[$row])) break;
	    for($x = 0;$x < $map[$row];$x++){
		for($y = 0; $y < $map[$row-1]; $y++){
		    
		    $weight_id = $map[$row-1]*$x+$y;
		    if(!isset($layer_weights[$weight_id])){
			damn(-1,'bad map (w) data in file '.$map_name);
			return false; // bad data
		    }
		    $weights[$row][$x][$y] = $layer_weights[$weight_id];
		    $delta_weights[$row][$x][$y] = 0;
		    $last_gradients[$row][$x][$y] = 0;
		    $last_weight_change[$row][$x][$y] = 0;
		    $this->update_values[$row][$x][$y] = $this->initial_update;
		}
	    }
	}
	foreach($map as $id => $layer_neu_count)
	{
	    if(!$id) continue;
	    $row = $id+count($map)-1;
	    $bias_weights[$id] = explode(' ',trim(doublespace($map_data[$row])));
	    foreach($bias_weights[$id] as $bw_id=>$bw){
	        $bias_delta_weights[$id][$bw_id] = 0;
	        $last_bias_change[$id][$bw_id] = 0;
	        $this->bias_update_values[$id][$bw_id] = $this->initial_update;
	        $this->last_bias_gradients[$id][$bw_id] = 0;
	    }
	    if(count($bias_weights[$id]) != $map[$id]){
	        damn(-1,'bad map (bw) data in file '.$map_name);
	    }
	}
	$this->map = $map;
	$this->weights = $weights;
	$this->bias_weights = $bias_weights;
	$this->delta_weights = $delta_weights;
	$this->last_gradients = $last_gradients;
	$this->last_weight_change = $last_weight_change;
	$this->bias_delta_weights = $bias_delta_weights;
	$this->last_bias_change = $last_bias_change;
	$this->buildTerrain();
    }
    
    function write_map_file($map_name = 'newmap'){
	$f = fopen($map_name,'w');
	if(!$f) return;
	fwrite($f,implode(' ',$this->map)."\n");
	foreach($this->weights as $i => $layer)
	{
	    foreach($layer as $neu){
		fwrite($f, implode(' ',$neu).' ');
	    }
	    fwrite($f, "\n");
	}
	foreach($this->map as $i => $num)
	{
	    if(!$i) continue;
	    fwrite($f, implode(' ',$this->bias_weights[$i])."\n");
	}
    }
    
    
    function echo_map_file(){
	damn(5,'map='.implode(' ',$this->map)."\n");
	foreach($this->weights as $i => $layer)
	{
	    foreach($layer as $j => $neu){
		damn(15,'weights_on_layer['.$j.']='. implode(' ',$neu).' ');
	    }
	    damn(15, "\n");
	}
	foreach($this->map as $i => $num)
	{
	    if(!$i) continue;
	    damn(15, 'bias_weights_on_layer['.$i.']='.implode(' ',$this->bias_weights[$i])."\n");
	}
    }
    
    
    function buildTerrain(){
	unset($this->terrain);
	$this->terrain = array();
	foreach($this->map as $layer_id => $layer_neuron_count){
	    for($i=0;$i<$layer_neuron_count;$i++){
		$this->terrain[$layer_id][] = new CNeu();
	    }
	}
    }
    
    function dumpNetworkData(){
	damn(10,'map:'.print_r($this->map,true));
	damn(10,'bias_weights:'.print_r($this->bias_weights,true));
	damn(10,'terrain:'.print_r($this->terrain,true));
	damn(10,'weights:'.print_r($this->weights,true));
    }
    
    function input($input){
	foreach($this->terrain[0] as $neu_local_id => &$neu)
	    {
		if(isset($input[$neu_local_id])){
		    // $neu->poke($input[$neu_local_id]);
		    $neu->output = $input[$neu_local_id];
		} else {
		    // $neu->poke(0);
		    $neu->output = $input[$neu_local_id];
		}
	}
    }
    
    
    
    /*
	$input = array of inputs for neurons. count should be the same as map[0] but is flexible. if count< then input=0 will be assumed for the rest. if count> extra will be ignored duh
    */
    function run($input = array(1)){
	// damn(-1,'running network');
	// $this->dumpNetworkData();
	$layers = count($this->map);
	$prev_layer = false;
	$this->input($input);
	foreach($this->terrain as $id_layer => &$layer)
	{
	    damn(0,'running layer: '.$id_layer);
	    if(!$prev_layer){
		$prev_layer = $layer;
		continue;
	    }
	    foreach($layer as $id_neuron => $neuron){
		$sum = 0;
		foreach($prev_layer as $id_prev_neuron => $prev_neuron){
		    if(!isset($this->weights[$id_layer][$id_neuron][$id_prev_neuron])){
			damn(-1,'bad weight data. check map, terrain and weights');
			return false;
		    }
		    $sum += $prev_neuron->output*$this->weights[$id_layer][$id_neuron][$id_prev_neuron];
		}
		$strength = $sum + $this->bias_weights[$id_layer][$id_neuron];
		// damn(-1,'poke ['.$id_layer.','.$id_neuron.']');
		$neuron->poke($strength);
	    } unset($neuron);
	    $prev_layer = $layer;
	} unset($layer); unset($id_layer);
	$output = array();
	foreach($prev_layer as $neuron){
	    $output[] = $neuron->output;
	} unset($neuron); unset($prev_layer);
	return $output;
    }
    
    
    
    function compare_outputs($new_output,$old_output){
	foreach($old_output as $i => $val){
	    if(abs($val-$new_output[$i])>$this->precision){
		return false;
	    }
	}
	return true;
    }
    
    /*function update_value($i,$j,$k,$set_value = false){
	if($set_value === false) $set_value = $this->initial_update;
	if(!isset($this->update_values[$i])){
	    return $this->update_values[$i][$j][$k] = $set_value;
	}elseif(!isset($this->update_values[$i][$j])){
	    return $this->update_values[$i][$j][$k] = $set_value;
	}elseif(!isset($this->update_values[$i][$j][$k])){
	    return $this->initial_update[$i][$j][$k] = $set_value;
	}
	return  $this->initial_update[$i][$j][$k] = $set_value;
    }*/
    
    
    
    function rprop_train_step_0(){
	    damn(5,'Reset deltas and shit');
	    $this->train = false;
	    $this->deltas = array(); // as many as the neurons
	    $this->bias_deltas = array();
	    $this->errors = array(); // as many as the output neurons
	    $this->gradients = array(); // as many as the weights (ouch)
	    $this->bias_gradients = array();
	    // $this->sum_errors = 0;
	    // $this->last_sum_errors = 0;
    }
    
    function rprop_train_step_1($input = array(1), $expected = array(1), $good_map_name = 'good_map_0'){
	damn(1,'training');
	//while($train)
	{
	    $this->last_sum_errors = $this->sum_errors;
	    $this->sum_errors = 0;
	    $this->echo_map_file();
	    $output = $this->run($input);
	    $similar = $this->compare_outputs($output,$this->last_output);
	    $this->last_output = $output;
	    damn(1,'output: '.implode(' ',$output));
	    damn(1,'similar: '.$similar);
	    
	    $l = count($this->map)-1;
	    $i = $this->map[$l]; // output layer
	    if(!isset($this->deltas[$l])) $this->deltas[$l] = array();
	    if(count($output) != $i) return false;
	    
	    for($k = 0;$k < $i;$k++){
		if(!isset($this->errors[$k])) $this->errors[$k] = 0;
	        $this->errors[$k] = $this->terrain[$l][$k]->output - $expected[$k];
	        if(abs($this->errors[$k]) > $this->precision){
	    	    damn(1,'error['.$k.']>'.$this->precision);
	    	     $this->train = true;
		}
	        $this->sum_errors += abs($this->errors[$k]);
	        if(!isset($this->deltas[$l][$k])) $this->deltas[$l][$k] = 0;
		if(!isset($this->bias_deltas[$l][$k])) $this->bias_deltas[$l][$k] = 0;
		$this->deltas[$l][$k] = $this->errors[$k] * activation_D($this->terrain[$l][$k]->strength);
		$this->bias_deltas[$l][$k] = $this->errors[$k] * activation_D(1);
	        //$deltas[$l][$k] = $errors[$k] * activation_D($this->terrain[$l][$k]->strength);
	    }
	    
	    $this->mean_error = $this->sum_errors/$i;
	    
	    
	    // if(!$train) return false;
	    
	    damn(1,'expected: '.implode(' ',$expected));
	    damn(1,'mean_error: '.$this->mean_error);
	    damn(1,'errors: '.implode(' ',$this->errors));
	    damn(1,'deltas: '.implode(' ',$this->deltas[$l]));
	    damn(1,'bias_deltas: '.implode(' ',$this->bias_deltas[$l]));
	    damn(2,'o= '.implode(' ',$output).' [s='.($similar?'1':'0').']; exp='.implode(' ',$expected).'; err='.implode(' ',$this->errors).'; dltk='.implode(' ',$this->deltas[$l]).';;');
	    $l--;
	    for(;$l>0;$l--){ // go backwards through layers
		for($k = 0;$k<$this->map[$l];$k++){ // for each neuron in the current layer
		    $w_sum = 0;
		    $bw_sum = 0;
		    for($j=0;$j<$this->map[$l+1];$j++){ // for each neuron in the next superior layer
			$w_sum += $this->weights[$l+1][$j][$k]*$this->deltas[$l+1][$j]; // sum of ((weights that goes out from current neuron $k to each neuron $j in the next layer) times previous deltas)
			$bw_sum += $this->bias_weights[$l+1][$j] * $this->bias_deltas[$l+1][$j];
		    }
		    damn(5,'w_sum['.$l.']['.$k.']='.$w_sum);
		    
		    if(!isset($this->deltas[$l][$k])) $this->deltas[$l][$k] = 0;
		    $this->deltas[$l][$k] = $w_sum * activation_D($this->terrain[$l][$k]->strength); // new delta is f' times w_sum       f' = derivative of f
		    
		    if(!isset($this->bias_deltas[$l][$k])) $this->bias_deltas[$l][$k] = 0;
		    damn(5,'bw_sum['.$l.']['.$k.']='.$bw_sum);
		    $this->bias_deltas[$l][$k] = $bw_sum * activation_D(1);
		    
		}
		
		damn(5,'hidden_deltas['.$l.']='.implode(', ',$this->deltas[$l]));
		damn(5,'hidden_bias_deltas['.$l.']='.implode(', ',$this->bias_deltas[$l]));
	    }
	    
	    
	    
	    
	    for($l = count($this->map)-1;$l>0;$l--){
		damn(5,'collecting gradients on layer '.$l);
		damn(5,'neurons_on_layer='.$this->map[$l].'; neurons_on_prev_layer='.$this->map[$l-1].';');
		if(!isset($this->gradients[$l])) $this->gradients[$l]= array();
		for($k = 0;$k<$this->map[$l];$k++){ // pentru fiecare neuron de pe layer L
		    if(!isset($this->gradients[$l][$k])) $this->gradients[$l][$k] = array(); // initializez derivatele
		    // damn(5,'current_weights_of_neuron['.$l.']['.$k.']='.implode(', ',$this->weights[$l][$k]));
		    for($j=0;$j<$this->map[$l-1];$j++){ // pentru fiecare neuron de pe layer L-1 (mai in interior, inclusiv input layer)
			if(!isset($this->gradients[$l][$k][$j])) $this->gradients[$l][$k][$j] = 0;
			$this->gradients[$l][$k][$j] += $this->terrain[$l-1][$j]->output * $this->deltas[$l][$k]; // derivata
		    }
		    if(!isset($this->bias_gradients[$l][$k])) $this->bias_gradients[$l][$k] = 0;
		    $this->bias_gradients[$l][$k] += $this->bias_deltas[$l][$k];
		}
	    }
	    
	}
	return $this->train;
    }
    
    function rprop_train_step_2($good_map_name = 'good_xor_00'){
	$delta = array();
	    for($l = count($this->map)-1;$l>0;$l--){
		// $delta[$l] = array();
		damn(5,'adjusting weights on layer '.$l);
		damn(5,'neurons_on_layer='.$this->map[$l].'; neurons_on_prev_layer='.$this->map[$l-1].';');
		for($k = 0;$k<$this->map[$l];$k++){ // pentru fiecare neuron de pe layer L
		    // $delta[$l][$k] = array();
		    damn(5,'current_weights_of_neuron['.$l.']['.$k.']='.implode(', ',$this->weights[$l][$k]));
		    damn(5,'current_bias_weight_for_neuron['.$l.']['.$k.']='.implode(', ',array($this->bias_weights[$l][$k])));
		    for($j=0;$j<$this->map[$l-1];$j++){ // pentru fiecare neuron de pe layer L-1 (mai in interior, inclusiv input layer)
			// sign of current gradient
			$gradient_sign = 1;
			if($this->gradients[$l][$k][$j]) $gradient_sign = $this->gradients[$l][$k][$j] / abs($this->gradients[$l][$k][$j]);
			// sign change for gradient
			$change = $this->gradients[$l][$k][$j] * $this->last_gradients[$l][$k][$j];
			damn(5,'change: '.$change.'; gradient_sign: '.$gradient_sign.'; gradient: '.$this->gradients[$l][$k][$j].'; last_gradient: '.$this->last_gradients[$l][$k][$j]);
			// save current gradient
			if($change > 0){
			    // $delta_weights[$l][$k][$j] = min($this->deltas[$l][$k]*$this->eta_positive,$this->max_delta_step);
			    $this->update_values[$l][$k][$j] = min($this->update_values[$l][$k][$j] * $this->eta_positive,$this->max_delta_step);
			    $weightChange = -1 * $gradient_sign * $this->update_values[$l][$k][$j];
			    $this->last_gradients[$l][$k][$j] = $this->gradients[$l][$k][$j];
			    
			    // $bias_delta_weights[$l][$k] = min($this->bias_deltas[$l][$k]*$this->eta_positive,$this->max_delta_step);
			    
			}else
			if($change < 0){
			    // $delta_weights[$l][$k][$j] = max($this->deltas[$l][$k]*$this->eta_negative,$this->min_delta_step);
			    $this->update_values[$l][$k][$j] = max($this->update_values[$l][$k][$j] * $this->eta_negative,$this->min_delta_step);
			    // $bias_delta_weights[$l][$k] = max($this->bias_deltas[$l][$k]*$this->eta_negative,$this->min_delta_step);
			    
			    //$weightChange = - $this->last_weight_change[$l][$k][$j];
			    //if($this->sum_errors > $this->last_sum_errors){
				$weightChange = - $this->last_weight_change[$l][$k][$j]; // go back
				// $bias_weightChange = - $this->last_bias_change[$l][$k];
			    //} else {
			//	$weightChange = -1 * $gradient_sign * $this->update_values[$l][$k][$j];
				//
			//    }
			    // $this->last_gradients[$l][$k][$j] = $this->gradients[$l][$k][$j];
			    $this->last_gradients[$l][$k][$j] = 0;
			}else
			if($change == 0){
			    $weightChange = -1 * $gradient_sign * $this->update_values[$l][$k][$j];
			    $this->last_gradients[$l][$k][$j] = $this->gradients[$l][$k][$j];
			    
			    
			}
			// $this->last_gradients[$l][$k][$j] = $this->gradients[$l][$k][$j];
			$this->weights[$l][$k][$j] += $weightChange;
			$this->last_weight_change[$l][$k][$j] = $weightChange;
			
		    }
		    
		    $bias_gradient_sign = 1;
		    
		    
		    if($this->bias_gradients[$l][$k]) $bias_gradient_sign = $this->bias_gradients[$l][$k] / abs($this->bias_gradients[$l][$k]);
		    if(!isset($this->last_bias_gradients[$l][$k])) $this->last_bias_gradients[$l][$k] = 0;
		    $bias_change = $this->bias_gradients[$l][$k] * $this->last_bias_gradients[$l][$k];
		    damn(5,'bias_change: '.$bias_change.'; bias_gradient_sign: '.$bias_gradient_sign.'; bias_gradient: '.$this->bias_gradients[$l][$k].'; bias_last_gradient: '.$this->last_bias_gradients[$l][$k]);
		    if($bias_change > 0){
			// $bias_delta_weights[$l][$k] = min($this->bias_deltas[$l][$k]*$this->eta_positive,$this->max_delta_step);
			// $bias_weightChange = -1 * $bias_gradient_sign * $bias_delta_weights[$l][$k];
			$this->bias_update_values[$l][$k] = min($this->bias_update_values[$l][$k] * $this->eta_positive, $this->max_delta_step);
			$bias_weightChange = -1 * $gradient_sign * $this->bias_update_values[$l][$k];
			$this->last_bias_gradients[$l][$k] = $this->bias_gradients[$l][$k];
		    }else
		    if($bias_change < 0){
			// $bias_delta_weights[$l][$k] = max($this->bias_deltas[$l][$k]*$this->eta_negative,$this->min_delta_step);
			$this->bias_update_values[$l][$k] = max($this->bias_update_values[$l][$k] * $this->eta_negative, $this->min_delta_step);
			// if($this->sum_errors > $this->last_sum_errors){
				$bias_weightChange = - $this->last_bias_change[$l][$k]; // go back
			//    } else {
				// $bias_weightChange = -1 * $bias_gradient_sign * $bias_delta_weights[$l][$k];
			//	$bias_weightChange = -1 * $gradient_sign * $this->bias_update_values[$l][$k];
			//    }
			$this->last_bias_gradients[$l][$k] = 0;
		    }else
		    if($bias_change == 0){
			// $bias_weightChange = -1 * $bias_gradient_sign * $this->bias_deltas[$l][$k];
			$bias_weightChange = -1 * $gradient_sign * $this->bias_update_values[$l][$k][$j];
			$this->last_bias_gradients[$l][$k] = $this->bias_gradients[$l][$k];
		    }
		    
		    $this->bias_weights[$l][$k] += $bias_weightChange;
		    $this->last_bias_change[$l][$k] = $bias_weightChange;
		    
		    damn(5,'adjusted_weights_of_neuron['.$l.']['.$k.']='.implode(', ',$this->weights[$l][$k]));
		    damn(5,'adjusted_bias_weight_for_neuron['.$l.']['.$k.']='.implode(', ',array($this->bias_weights[$l][$k])));
		}
	}
	$this->write_map_file($good_map_name);
    }
    
    
    
    
    
    
    
    
    
}
?>