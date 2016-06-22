<?php
$handle = fopen('publications-v8.txt', "r");

if ($handle) {
    while (($line = fgets($handle)) !== false) {
        $line = trim($line);
        if (substr($line, 0, 2) == "#*") {
            echo substr($line, 2);
        } elseif (substr($line, 0, 2) == "#!") {
            echo " ";
            echo substr($line, 2);
        } elseif ($line == "") {
            echo "\n";
        }
    }
    fclose($handle);
} else {
    die("Couldn't open file...");
}
?>
