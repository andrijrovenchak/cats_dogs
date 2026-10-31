###### Cats&Dogs pipeline: ######

# 1. Collect FASTA
python collect_fastas.py

# 1a. Do manual lookup for species with empty FASTA sets like
# Leopardus emiliae -> []
# or
# Vulpes bengalensis -> []
# E.g., check literature, look for other possible species names, etc.
# Consider searching almost complete mitochondrions as well if applicable.

# 2. Remove shared FASTA. 
# These might occur due to species/subspecies ambiguity, like
# Canis lupus familiaris -- treating domestic dog as a subspecies -- will end up in both Canis lupus and Canis familiaris
python remove_shared_records.py felidae_fastas felidae_clean
python remove_shared_records.py canidae_fastas canidae_clean

# 2a. Remove duplicated FASTA
python deduplicate_fastas.py felidae_clean felidae_dedup --report felidae_dedup_report.csv
python deduplicate_fastas.py canidae_clean canidae_dedup --report canidae_dedup_report.csv

# 3. Provide summary. Prepare tables.