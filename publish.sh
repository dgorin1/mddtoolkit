# !/bin/bash

# delete the dist folder
rm -rf dist

# if the next version is not specified, figure it out using the version in the package
VERSION=$(python3 -c "import diffusion_kinetics; print(diffusion_kinetics.__version__)")
if [ -z "$1" ]
then
    # get the package name from package version
    # iterate the version
    NEXTVERSION=$(echo ${VERSION} | awk -F. -v OFS=. '{$NF += 1 ; print}')
else
    NEXTVERSION=$1
fi

# show the next version and confirm with the user
echo "Current version: ${VERSION}"
read -p "Next version: ${NEXTVERSION}. Continue? (y/n) " -n 1 -r

# if the user confirms, publish the package
if [[ $REPLY =~ ^[Yy]$ ]]
then
    # update the version in the package
    sed -i '' "s/${VERSION}/${NEXTVERSION}/g" src/diffusion_kinetics/__init__.py
    python3 -m build
    python3 -m twine upload dist/* \
        --username __token__ \
        --password ${PYPI_TOKEN}
fi

