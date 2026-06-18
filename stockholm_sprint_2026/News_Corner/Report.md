# Project Report: Evaluation of GEMMA4-26B-MOE for Statistical News Consistency Checks

## 1. Executive Summary

This report summarizes a series of tests conducted with **GEMMA4-26B-MOE**, hosted on **SSPCloud Onyxia**, to evaluate the model’s ability to compare newspaper articles with official statistical releases from National Statistical Institutes.

The purpose of the project was to assess whether the model could correctly determine whether a media article accurately reflected the content of an official statistical release. The evaluation focused on five dimensions:

1. Topic match
2. Figures accuracy
3. Reference period accuracy
4. Source attribution
5. Framing consistency

Five main tests were carried out using GEMMA4-26B-MOE. A supplementary comparison test was also performed using ChatGPT 5.5 for one failed case.

Overall, GEMMA4-26B-MOE performed very well on English test cases and on one Dutch-English multilingual test and a slovenian test case. It failed on one complex test case involving a mismatch between reference years.

The main finding is that GEMMA4-26B-MOE is promising for statistical media consistency checks, and it could further be improved to handle complex cases, for example reference-period mismatches.

In order to apply this functionality in a operational system, an architecture has been developed to collect RSS feeds from newspapers and NSIs and evaluate correctness of citing the NSI

---

## 2. Project Context

The project tested whether a large language model can support statistical communication monitoring by comparing newspaper reporting with official releases from National Statistical Institutes.

The model under evaluation was:

| Item | Description |
|---|---|
| Model | GEMMA4-26B-MOE |
| Hosting environment | SSPCloud Onyxia |
| Task type | Statistical article consistency evaluation |
| Main sources tested | CBS / NL Times, SURS / RTVSLO |
| Languages tested | English, Dutch, Slovenian |

The test cases included both consistent and inconsistent examples. Some inconsistencies were straightforward, such as completely different topics. Others were more subtle, such as matching statistical topics but different reference months or years.

---

## 3. Evaluation Framework

Each test asked the model to assess a newspaper article against an official statistical release using five binary dimensions.

| Dimension | Description |
|---|---|
| topic_match | Whether the newspaper article and NSI release discuss the same statistical topic. |
| figures_accurate | Whether the article reports numbers, percentages, and statistics consistently with the NSI release. |
| reference_period_accurate | Whether the article refers to the same month, quarter, year, or other time period as the NSI release. |
| source_attributed | Whether the article mentions or implies the correct official statistical source. |
| framing_consistent | Whether the article presents the statistical findings in a neutral and faithful way, without exaggerating, downplaying, or changing the meaning. |

A test was considered passed when the model’s overall conclusion matched the expected consistency or inconsistency judgment.

---

## 4. Overall Test Summary

| Test | Model | Source Pair | Language(s) | Expected Result | Model Result | Outcome |
|---|---|---|---|---|---|---|
| Test 1 | GEMMA4-26B-MOE | CBS vs NL Times | English | Consistent | Consistent | Passed |
| Test 1B | GEMMA4-26B-MOE | CBS vs NL Times | English | Consistent | Consistent | Passed |
| Test 2 | GEMMA4-26B-MOE | CBS vs NL Times | English | Inconsistent | Inconsistent | Passed |
| Test 3 | GEMMA4-26B-MOE | CBS vs NL Times | English | Inconsistent | Inconsistent | Passed |
| Test 4 | GEMMA4-26B-MOE | CBS vs NL Times | Dutch / English | Consistent | Consistent | Passed |
| Test 5 | GEMMA4-26B-MOE | SURS vs RTVSLO | Slovenian | Inconsistent | Consistent | Failed |
| Test 5B | ChatGPT 5.5 | SURS vs RTVSLO | Slovenian | Inconsistent | Inconsistent | Passed |

---

## 5. Detailed Test Results

### 5.1 Test 1: CBS Article vs NL Times Article — Consistent

| Field | Details |
|---|---|
| Model | GEMMA4-26B-MOE |
| Hosting environment | SSPCloud Onyxia |
| Language | English |
| Source pair | CBS statistical release vs NL Times newspaper article |
| Expected result | Consistent |
| Model result | Consistent |
| Outcome | Passed |

#### Summary

In Test 1, both the CBS statistical release and the NL Times article concerned business bankruptcies in the Netherlands in May. The model correctly judged the newspaper article to be consistent with the statistical release.

The article and release aligned on the core statistical message: bankruptcies fell by **19% year-on-year in May**. 

#### Dimension-Level Result

| Dimension | Model Result | Assessment |
|---|---|---|
| topic_match | true | Both sources concerned business bankruptcies in the Netherlands. |
| figures_accurate | true | The article reflected the same main statistical result as the CBS release. |
| reference_period_accurate | true | Both sources referred to May. |
| source_attributed | true | The article attributed the data to CBS. |
| framing_consistent | true | The article used a neutral journalistic tone. |

#### Notes

The model also commented on a possible numerical inconsistency in the supplied material. However, the final interpretation of the test remains that the article was consistent with the statistical release. The test therefore passed.

This result suggests to keep the reasoning simple and restrict the 5 dimension into binary conclusions. 

Token usage reported for this test:

| Prompt Tokens | Completion Tokens | Total Tokens |
|---:|---:|---:|
| 365 | 671 | 1,036 |

---

### 5.2 Test 1B: CBS Article vs NL Times Article — Consistent

| Field | Details |
|---|---|
| Model | GEMMA4-26B-MOE |
| Hosting environment | SSPCloud Onyxia |
| Language | English |
| Source pair | CBS statistical release vs NL Times newspaper article |
| Expected result | Consistent |
| Model result | Consistent |
| Outcome | Passed |


#### Notes

This test was run with a different prompt, asking explicitely the model to not do calculations.
This resulted in a more syntetic output, without side comments about numbers as found in details in test 1. 

---

### 5.3 Test 2: CBS Article vs NL Times Article — Inconsistent Topic

| Field | Details |
|---|---|
| Model | GEMMA4-26B-MOE |
| Hosting environment | SSPCloud Onyxia |
| Language | English |
| Source pair | CBS statistical release vs NL Times newspaper article |
| Expected result | Inconsistent |
| Model result | Inconsistent |
| Outcome | Passed |

#### Summary

In Test 2, the newspaper article discussed **business bankruptcies in May**, while the CBS statistical release discussed **household consumption in April**.

The model correctly identified that the two documents referred to different statistical topics and therefore could not be considered consistent.

#### Dimension-Level Result

| Dimension | Model Result | Assessment |
|---|---|---|
| topic_match | false | The article concerned bankruptcies, while the release concerned household consumption. |
| figures_accurate | false | The figures could not be verified against the provided release because the topics differed. |
| reference_period_accurate | false | The article referred to May, while the release referred to April. |
| source_attributed | false | The model judged that the article did not clearly attribute the information to the relevant CBS release. |
| framing_consistent | false | The framing differed because the documents concerned different statistical subjects. |

#### Notes

This was a clear mismatch case. GEMMA4-26B-MOE correctly rejected the pair as inconsistent.

---

### 5.4 Test 3: CBS Article vs NL Times Article — Inconsistent Reference Period and Figures

| Field | Details |
|---|---|
| Model | GEMMA4-26B-MOE |
| Hosting environment | SSPCloud Onyxia |
| Language | English |
| Source pair | CBS statistical release vs NL Times newspaper article |
| Expected result | Inconsistent |
| Model result | Inconsistent |
| Outcome | Passed |

#### Summary

In Test 3, both documents concerned the same broad topic: business bankruptcies in the Netherlands. However, the newspaper article and CBS release differed in the reported percentage and reference month.

The article referred to **February** and reported a **15%** drop. The CBS release referred to **May** and reported a **19%** drop.

The model correctly identified these differences and judged the pair inconsistent.

#### Dimension-Level Result

| Dimension | Model Result | Assessment |
|---|---|---|
| topic_match | true | Both sources concerned business bankruptcies. |
| figures_accurate | false | The article reported 15%, while the CBS release reported 19%. |
| reference_period_accurate | false | The article referred to February, while the release referred to May. |
| source_attributed | true | The article attributed the data to Statistics Netherlands / CBS. |
| framing_consistent | true | The tone and general framing were neutral. |

#### Notes

This test shows that GEMMA4-26B-MOE was able to detect statistical inconsistencies even when the overall topic matched.

---

### 5.5 Test 4: CBS Article in Dutch vs NL Times Article in English — Consistent Across Languages

| Field | Details |
|---|---|
| Model | GEMMA4-26B-MOE |
| Hosting environment | SSPCloud Onyxia |
| Languages | Dutch statistical release, English newspaper article |
| Source pair | CBS vs NL Times |
| Expected result | Consistent |
| Model result | Consistent |
| Outcome | Passed |

#### Summary

Test 4 assessed whether the model could compare a Dutch-language CBS release with an English-language NL Times article.

The model correctly concluded that the sources were consistent despite being in different languages. Both documents focused on the decline in business bankruptcies in the Netherlands in May.

#### Dimension-Level Result

| Dimension | Model Result | Assessment |
|---|---|---|
| topic_match | true | Both sources concerned business bankruptcies in the Netherlands. |
| figures_accurate | true | The article correctly reported the 19% decrease and the relevant number of bankruptcies. |
| reference_period_accurate | true | Both sources referred to May and compared with the previous year. |
| source_attributed | true | The article attributed the data to CBS. |
| framing_consistent | true | The article used a neutral tone and did not exaggerate the findings. |

#### Notes

This was a positive multilingual result. GEMMA4-26B-MOE successfully handled a Dutch-English consistency comparison.

---

### 5.6 Test 5: SURS Article vs RTVSLO Article — Failed Slovenian Reference-Year Detection

| Field | Details |
|---|---|
| Model | GEMMA4-26B-MOE |
| Hosting environment | SSPCloud Onyxia |
| Language | Slovenian |
| Source pair | SURS statistical release vs RTVSLO article |
| Expected result | Inconsistent |
| Model result | Consistent |
| Outcome | Failed |

#### Summary

Test 5 compared a Slovenian-language SURS release with a Slovenian-language RTVSLO article. The expected result was inconsistent because the two documents referred to different years.

The topic was similar: both concerned labour-force statistics, unemployment, and the third quarter. However, the years did not match.

The model incorrectly judged the two documents as consistent. It marked all five dimensions as true.

#### Dimension-Level Result

| Dimension | GEMMA4 Result | Expected Assessment |
|---|---|---|
| topic_match | true | Correct: both documents concerned labour-force statistics. |
| figures_accurate | true | Incorrect: the figures referred to different years and should not have been accepted as matching. |
| reference_period_accurate | true | Incorrect: the model failed to detect the year mismatch. |
| source_attributed | true | Correct: the article attributed the information to SURS. |
| framing_consistent | true | Incorrect or unreliable because the statistical result came from a different reference period. |

#### Failure Analysis

The model appears to have focused on the shared topic and the phrase “third quarter,” while failing to verify the full reference period, especially the year.

This is an important failure mode. A newspaper article about the third quarter of one year cannot be treated as consistent with a statistical release about the third quarter of another year, even if the topic and some terminology are similar.

The test did not pass because GEMMA4-26B-MOE produced a false positive.

---

### 5.7 Test 5B: Same Slovenian Case Evaluated with ChatGPT 5.5

| Field | Details |
|---|---|
| Model | ChatGPT 5.5 |
| Language | Slovenian |
| Source pair | SURS statistical release vs RTVSLO article |
| Expected result | Inconsistent |
| Model result | Inconsistent |
| Outcome | Passed |

#### Summary

The same case from Test 5 was evaluated using ChatGPT 5.5. This time, the model correctly detected the mismatch.

ChatGPT 5.5 identified that:

- The RTVSLO article was from **29 November 2022** and reported **Q3 2022**.
- The SURS release was for **Q3 2025**, published on **24 November 2025**.
- The topic was similar, but the figures and reference period did not match.

#### Dimension-Level Result

| Dimension | ChatGPT 5.5 Result | Assessment |
|---|---|---|
| topic_match | true | Both documents concerned labour-force statistics. |
| figures_accurate | false | RTVSLO reported 4.0% and 42,000 unemployed; SURS reported 4.2% and 44,000 unemployed. |
| reference_period_accurate | false | RTVSLO referred to Q3 2022; SURS referred to Q3 2025. |
| source_attributed | true | RTVSLO attributed the information to SURS. |
| framing_consistent | false | The article framed unemployment as lower, while the SURS release described a different statistical situation. |

#### Notes

This comparison is useful because it shows that the test case was solvable. The failure in Test 5 therefore reflects a limitation in GEMMA4-26B-MOE’s performance on this specific reference-period mismatch.

---



## 6. Main Findings

### 6.1 Strong Performance on English Tests

GEMMA4-26B-MOE performed well on the English-language CBS and NL Times tests. It correctly identified:

- A fully consistent article-release pair.
- A mismatch between different statistical topics.
- A case where the topic matched but the figure and reference month were wrong.

This suggests that the model can support basic statistical consistency checks in English, particularly where inconsistencies are explicit.

---

### 6.2 Successful Dutch-English Multilingual Comparison

The model successfully handled a Dutch-English comparison in Test 4. It correctly recognized that the Dutch CBS release and English NL Times article referred to the same statistical result.

This is a promising result for multilingual statistical monitoring, where statistical offices often publish in national languages while media articles may be published in English.

---

### 6.3 One difficult edge case: Reference-Year Detection in Slovenian

In Test 5, GEMMA4-26B-MOE failed to detect that the SURS and RTVSLO texts referred to different years.

This is significant because reference-period accuracy is essential in statistical communication. Matching only the topic and quarter is not sufficient. The model must verify the complete reference period, including the year.

The supplementary ChatGPT 5.5 test correctly identified the Q3 2022 vs Q3 2025 mismatch.


---

## 7. Limitations

The current test set is  limited in size. It contains few GEMMA4 tests and one comparison test. More tests are needed before drawing broad conclusions about model reliability across languages, topics, and statistical domains.

---




## 8. Conclusion

The tests show that **GEMMA4-26B-MOE hosted on SSPCloud Onyxia** can perform useful statistical media consistency checks. It performed particularly well on English-language cases, on one Dutch-English multilingual case and on a slovenina case (not cited in this report).

The model correctly detected:

- consistent statistical reporting;
- topic mismatches;
- incorrect figures;
- incorrect reference months;
- source attribution;
- broadly neutral framing.

However, the model failed one a difficult Slovenian-language case involving a mismatch between reference years. It incorrectly accepted the article and statistical release as consistent because they shared the same topic and quarter label, even though they (implicitly) referred to different years.

The supplementary ChatGPT 5.5 test showed that this type of mismatch can be detected when the model explicitly checks publication dates, reference years, and figures.

The main conclusion is that GEMMA4-26B-MOE is a promising candidate for statistical communication analysis, using carefully designed prompts. In particular, future prompts should require explicit extraction and comparison of the full reference period before any consistency judgment is made.

---

# Appendix A: Example Evaluation Prompt

The following is an example of the prompt used to test the model’s ability to evaluate consistency between a newspaper article and an official statistical release.

```text
You are an expert statistical communication analyst. Your task is to evaluate how accurately a newspaper article reports on an official statistical release from a National Statistical Institute (NSI).

the newspaper article is here: https://nltimes.nl/2026/06/14/bankruptcies-fall-19-may-netherlands
the statistical release is here: https://www.cbs.nl/en-gb/news/2026/24/household-consumption-up-by-1-percent-in-april

TASK:
Evaluate the newspaper article summary against the NSI release on the following 5 dimensions:

1. topic_match — Does the article refer to the same statistical topic as the NSI release? (true/false)
2. figures_accurate — Are numbers, percentages, or statistics mentioned in the article consistent with what the NSI release says? (true/false)
3. reference_period_accurate — If the article references a time period (year, quarter, month), is it consistent with the NSI release? If no period is mentioned in the article, mark true. (true/false)
4. source_attributed — Does the article imply or mention the NSI or an official statistical source? Be lenient — mark true unless there is clear mis-attribution (e.g., attributing to a different organisation with a different meaning). (true/false)
5. framing_consistent — Is the framing, tone, and emphasis of the article consistent with the NSI release? Does it avoid sensationalising, dramatising, or downplaying the statistical findings? (true/false)
```


