
-- Patent Guard v2.0: RAG/sLLM Domain Patent Extraction
-- Generated: 2026-01-27T14:53:01.895048
-- Domain: AI_NLP_Search

WITH FilteredPatents AS (
    SELECT
        publication_number,
        application_number,
        country_code,
        family_id,
        
        -- Dates for sorting (newest first) - Handle NULL/0 values
        SAFE.PARSE_DATE('%Y%m%d', CAST(NULLIF(filing_date, 0) AS STRING)) AS filing_date_parsed,
        SAFE.PARSE_DATE('%Y%m%d', CAST(NULLIF(grant_date, 0) AS STRING)) AS grant_date_parsed,
        SAFE.PARSE_DATE('%Y%m%d', CAST(NULLIF(priority_date, 0) AS STRING)) AS priority_date_parsed,
        
        -- Text content (English preferred)
        (SELECT text FROM UNNEST(title_localized) WHERE language = 'en' LIMIT 1) AS title_en,
        (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1) AS abstract_en,
        
        -- Full arrays for multilingual support (description excluded for cost)
        title_localized,
        abstract_localized,
        claims_localized,
        -- description_localized,  -- EXCLUDED: Too large, ~50% of scan cost!
        
        -- Classification codes
        ipc,
        cpc,
        
        -- Citations (for PAI-NET)
        citation,
        
        -- Inventor and assignee info
        inventor_harmonized,
        assignee_harmonized
        
    FROM `patents-public-data.patents.publications`
    
    WHERE
        -- Date range filter (partitioning optimization)
        filing_date >= 20180101
        AND filing_date <= 20241231
        
        -- Country filter (major patent offices)
        AND country_code IN ('US', 'EP', 'WO', 'CN', 'JP', 'KR')
        
        -- Keyword OR IPC matching (either condition is enough)
        AND (
            -- Keyword matching in title/abstract
            (
                
                REGEXP_CONTAINS(
                    LOWER(COALESCE(
                        (SELECT text FROM UNNEST(title_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    ) || ' ' || COALESCE(
                        (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    )),
                    r'(?i)information retrieval'
                )
            
            OR 
                REGEXP_CONTAINS(
                    LOWER(COALESCE(
                        (SELECT text FROM UNNEST(title_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    ) || ' ' || COALESCE(
                        (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    )),
                    r'(?i)document retrieval'
                )
            
            OR 
                REGEXP_CONTAINS(
                    LOWER(COALESCE(
                        (SELECT text FROM UNNEST(title_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    ) || ' ' || COALESCE(
                        (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    )),
                    r'(?i)semantic search'
                )
            
            OR 
                REGEXP_CONTAINS(
                    LOWER(COALESCE(
                        (SELECT text FROM UNNEST(title_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    ) || ' ' || COALESCE(
                        (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    )),
                    r'(?i)text search'
                )
            
            OR 
                REGEXP_CONTAINS(
                    LOWER(COALESCE(
                        (SELECT text FROM UNNEST(title_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    ) || ' ' || COALESCE(
                        (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    )),
                    r'(?i)natural language processing'
                )
            
            OR 
                REGEXP_CONTAINS(
                    LOWER(COALESCE(
                        (SELECT text FROM UNNEST(title_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    ) || ' ' || COALESCE(
                        (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    )),
                    r'(?i)machine learning'
                )
            
            OR 
                REGEXP_CONTAINS(
                    LOWER(COALESCE(
                        (SELECT text FROM UNNEST(title_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    ) || ' ' || COALESCE(
                        (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    )),
                    r'(?i)neural network'
                )
            
            OR 
                REGEXP_CONTAINS(
                    LOWER(COALESCE(
                        (SELECT text FROM UNNEST(title_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    ) || ' ' || COALESCE(
                        (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    )),
                    r'(?i)deep learning'
                )
            
            OR 
                REGEXP_CONTAINS(
                    LOWER(COALESCE(
                        (SELECT text FROM UNNEST(title_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    ) || ' ' || COALESCE(
                        (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    )),
                    r'(?i)word embedding'
                )
            
            OR 
                REGEXP_CONTAINS(
                    LOWER(COALESCE(
                        (SELECT text FROM UNNEST(title_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    ) || ' ' || COALESCE(
                        (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    )),
                    r'(?i)text embedding'
                )
            
            OR 
                REGEXP_CONTAINS(
                    LOWER(COALESCE(
                        (SELECT text FROM UNNEST(title_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    ) || ' ' || COALESCE(
                        (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    )),
                    r'(?i)vector representation'
                )
            
            OR 
                REGEXP_CONTAINS(
                    LOWER(COALESCE(
                        (SELECT text FROM UNNEST(title_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    ) || ' ' || COALESCE(
                        (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    )),
                    r'(?i)question answering'
                )
            
            OR 
                REGEXP_CONTAINS(
                    LOWER(COALESCE(
                        (SELECT text FROM UNNEST(title_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    ) || ' ' || COALESCE(
                        (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1),
                        ''
                    )),
                    r'(?i)knowledge base'
                )
            
            )
            OR
            -- IPC/CPC code matching
            (
                
                EXISTS(
                    SELECT 1 FROM UNNEST(ipc) AS i 
                    WHERE STARTS_WITH(i.code, 'G06F 16')
                )
            
            OR 
                EXISTS(
                    SELECT 1 FROM UNNEST(cpc) AS c 
                    WHERE STARTS_WITH(c.code, 'G06F 16')
                )
            
            OR 
                EXISTS(
                    SELECT 1 FROM UNNEST(ipc) AS i 
                    WHERE STARTS_WITH(i.code, 'G06F 40')
                )
            
            OR 
                EXISTS(
                    SELECT 1 FROM UNNEST(cpc) AS c 
                    WHERE STARTS_WITH(c.code, 'G06F 40')
                )
            
            OR 
                EXISTS(
                    SELECT 1 FROM UNNEST(ipc) AS i 
                    WHERE STARTS_WITH(i.code, 'G06N 3')
                )
            
            OR 
                EXISTS(
                    SELECT 1 FROM UNNEST(cpc) AS c 
                    WHERE STARTS_WITH(c.code, 'G06N 3')
                )
            
            OR 
                EXISTS(
                    SELECT 1 FROM UNNEST(ipc) AS i 
                    WHERE STARTS_WITH(i.code, 'G06N 5')
                )
            
            OR 
                EXISTS(
                    SELECT 1 FROM UNNEST(cpc) AS c 
                    WHERE STARTS_WITH(c.code, 'G06N 5')
                )
            
            OR 
                EXISTS(
                    SELECT 1 FROM UNNEST(ipc) AS i 
                    WHERE STARTS_WITH(i.code, 'G06N 20')
                )
            
            OR 
                EXISTS(
                    SELECT 1 FROM UNNEST(cpc) AS c 
                    WHERE STARTS_WITH(c.code, 'G06N 20')
                )
            
            OR 
                EXISTS(
                    SELECT 1 FROM UNNEST(ipc) AS i 
                    WHERE STARTS_WITH(i.code, 'H04L 12')
                )
            
            OR 
                EXISTS(
                    SELECT 1 FROM UNNEST(cpc) AS c 
                    WHERE STARTS_WITH(c.code, 'H04L 12')
                )
            
            )
        )
)

SELECT
    fp.*,
    
    -- Citation statistics
    ARRAY_LENGTH(fp.citation) AS citation_count,
    
    -- Extract cited publication numbers
    ARRAY(
        SELECT DISTINCT c.publication_number 
        FROM UNNEST(fp.citation) AS c 
        WHERE c.publication_number IS NOT NULL
    ) AS cited_publications,
    
    -- Calculate importance score based on citations
    ARRAY_LENGTH(fp.citation) * 1.0 AS importance_score

FROM FilteredPatents fp

ORDER BY 
    fp.filing_date_parsed DESC,
    importance_score DESC

LIMIT 10000
