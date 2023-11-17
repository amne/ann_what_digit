<?php
define('_CONSOLE_LEVEL',-1);
global $debug_level;
$debug_level = 0;
function damn($level,$msg,$skipnl = false,$raw=false,$logfile = '.debug_info')
	{
	    global $debug_level;
	    if($debug_level < $level)
		return;
	
	    $amne = @fopen($logfile,'a');
	    if(!$amne) return;
	    $debuginfo = debug_backtrace();
	    if(!$skipnl) $msg.="\n";
	    if(!$raw) $msg = '('.str_replace($_SERVER['DOCUMENT_ROOT'],'.', $debuginfo[0]['file']).'::'.$debuginfo[0]['line'].') '.$msg;
	    if($level<_CONSOLE_LEVEL) echo $msg;
	    fwrite($amne,$msg);
	    fclose($amne);
	    return;
	}
?>