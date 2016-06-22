-- SQL commands to generate useful figures from the raw DBLP dataset

USE dblp-v8;

-- Add columns
-- ALTER TABLE authors ADD nb_papers INT(4) NULL,
--                     ADD min_citations INT(4) NULL,
--                     ADD avg_citations FLOAT(8) NULL;
-- ALTER TABLE papers ADD nb_citations INT(4) NULL;

-- Count the number of papers (with abstract) authored by user
-- SELECT pa.author AS a_id, COUNT(*) AS cnt_papers FROM paper_authors pa INNER JOIN papers p ON pa.paper = p.id WHERE p.abstract != "" GROUP BY a_id;

-- Compute the number of papers (with abstract) written by each author and update the column
UPDATE authors a INNER JOIN (
    SELECT pa.author AS a_id, COUNT(*) AS cnt_papers FROM paper_authors pa
    INNER JOIN papers p ON pa.paper = p.id
    WHERE p.abstract != ""
    GROUP BY a_id
) c ON c.a_id = a.id
SET a.nb_papers = c.cnt_papers;

-- Count the number of citations in a paper
-- SELECT c.cited_by AS c_id, COUNT(*) AS cnt_citations FROM citations c INNER JOIN papers p ON c.cited_paper = p.id WHERE p.abstract != "" GROUP BY c_id;

-- Compute the number of citations for each paper and update  the column
UPDATE papers p INNER JOIN (
    SELECT c.cited_by AS c_id, COUNT(*) AS cnt_citations FROM citations c
    INNER JOIN papers p ON c.cited_paper = p.id
    WHERE p.abstract != ""
    GROUP BY c_id
) cc ON cc.c_id = p.id
SET p.nb_citations = cc.cnt_citations;

-- Get the minimum number of citations in a paper among all the papers written by a user
-- SELECT pa.author AS a_id, MIN(nb_citations) AS c_min FROM papers p INNER JOIN paper_authors pa ON pa.paper = p.id WHERE p.abstract != '' GROUP BY a_id;
-- Get the average number of citations in a paper among all the papers written by a user
-- SELECT pa.author AS a_id, AVG(nb_citations) AS c_avg FROM papers p INNER JOIN paper_authors pa ON pa.paper = p.id WHERE p.abstract != '' GROUP BY a_id;

-- Compute the minimal and average numbers of citations in a paper for each author and update the columns
UPDATE authors a INNER JOIN (
    SELECT pa.author AS a_id, MIN(nb_citations) AS c_min, AVG(nb_citations) AS c_avg FROM papers p
    INNER JOIN paper_authors pa ON pa.paper = p.id
    WHERE p.abstract != ''
    GROUP BY a_id
) c ON c.a_id = a.id
SET a.min_citations = c.c_min,
    a.avg_citations = c.c_avg;
