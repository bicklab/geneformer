# This is a copy of Yash's code on Polaris.

Steps to copy files outlined below.

# Copy files to local computer
scp pershy1@polaris.alcf.anl.gov:/grand/GeomicVar/pershy1/scripts/\*.py "/Users/pershy1/geneformer/scripts/"
scp pershy1@polaris.alcf.anl.gov:/grand/GeomicVar/pershy1/scripts/batch_scripts/\*.sh "/Users/pershy1/geneformer/scripts/batch_scripts/"

# Push local computer to Github
cd /Users/pershy1/geneformer
git add .
git commit -m "Updating geneformer scripts" (replace with more descriptive if desired)
git push origin main

Now, polaris should match local which should match GitHub.
