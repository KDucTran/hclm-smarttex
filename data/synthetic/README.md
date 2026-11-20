# Synthetic smart-textile observations

Eight simulated subject streams supply 1,800 samples each at 1 Hz. The time column uses a common origin across the 14,400 rows. For each subject, part 1 contains the first 900 observations and part 2 contains the last 900.

| Subject | First 900 observations | Last 900 observations |
| ---: | --- | --- |
| 0 | [0–899 s](subject_00_part_1.csv) | [900–1,799 s](subject_00_part_2.csv) |
| 1 | [1,800–2,699 s](subject_01_part_1.csv) | [2,700–3,599 s](subject_01_part_2.csv) |
| 2 | [3,600–4,499 s](subject_02_part_1.csv) | [4,500–5,399 s](subject_02_part_2.csv) |
| 3 | [5,400–6,299 s](subject_03_part_1.csv) | [6,300–7,199 s](subject_03_part_2.csv) |
| 4 | [7,200–8,099 s](subject_04_part_1.csv) | [8,100–8,999 s](subject_04_part_2.csv) |
| 5 | [9,000–9,899 s](subject_05_part_1.csv) | [9,900–10,799 s](subject_05_part_2.csv) |
| 6 | [10,800–11,699 s](subject_06_part_1.csv) | [11,700–12,599 s](subject_06_part_2.csv) |
| 7 | [12,600–13,499 s](subject_07_part_1.csv) | [13,500–14,399 s](subject_07_part_2.csv) |

The files preserve the column order, decimal representations and row order of the original benchmark. The Python script verifies the SHA-256 of the complete CSV before extracting any features.

Input SHA-256: `e94144f93e6691d46e2b0b22525eee50df1b239114765d54ebfa0a5fd412c00c`
