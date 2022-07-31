#! /bin/sh
competition_name=$1

if [ "$competition_name" == "rm" ]
then
    rm -rf configs/$2
    rm -rf data/$2
    rm -rf metric/$2
    rm -rf models/$2
    rm -rf output/$2

    echo "Removed folders for competion: $2"
    exit 0
fi


mkdir configs/$competition_name
mkdir -p data/$competition_name/dataset
mkdir metric/$competition_name
mkdir models/$competition_name
mkdir output/$competition_name
echo "Created folders for competion: $competition_name"