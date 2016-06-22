<?php
$handle = fopen('bad_authors', "r");

if ($handle) {
    while (($line = trim(fgets($handle))) !== false && $line != "") {
        echo $line;
        echo "\n";
        $author = explode(' ', $line);
        $exp = '"/' . $author[0] . ' ' . $author[1] . '\\W/"';
        $papers = shell_exec("awk -v RS='' -v ORS='\\n\\n' ".$exp." ../publications-v8.txt");
        file_put_contents(strtolower($author[1]).'_papers.txt', $papers);
    }
    fclose($handle);
} else {
    die("Couldn't open file...");
}
?>
