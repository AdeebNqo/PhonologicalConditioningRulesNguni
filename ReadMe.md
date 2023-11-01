# Folders

- `/Data`: contains the raw datasets from SADILAR [1] and Ukwabelana [2].
- `/ProcessedData`: contains the cleaned up datasets
- `/Rules`: contains the extracted rules for each language
- `/Lib`: contains a collection of code to assist the extraction of rules.

# Extracting rules

The `ExtractRules.py` file can be used to extract rules. The command `Python ExtractRules.py` will create rules and save them in the `/Rules` folder. Progress of the extraction can be monitored via a logging file called `output.txt`.

# Reading the extracted rules

The `ReadRules.py` file can be used to read the extracted rules that are saved in the `/Rules/` folder. The command `python ReadRules.py` will produce results that look like the following:

    Language = xh
    In combination of wa-i-z-in-gingqi, we apply aizi->ee, and get weengingqi
    In combination of na-u-lu-phuhliso, we apply aulu->o, and get nophuhliso
    ...

# Citing the rules

Mahlaza, Z., Khumalo, L. Algorithm for assisting grammarians when extracting phonological conditioning rules for Nguni languages. Digital Humanities Association of Southern Africa (DHASA) 2023, November 27, 2023 to December 1, 2023, Nelson Mandela University

# References

1. https://repo.sadilar.org/handle/20.500.12185/546
2. Sebastian Spiegler, Andrew van der Spuy, and Peter A. Flach. 2010. Ukwabelana - an open-source morphological Zulu corpus. In COLING 2010, 23rd International Conference on Computational Linguistics, Proceedings of the Conference, 23-27 August 2010, Beijing, China, pages 1020–1028. Tsinghua University Press
