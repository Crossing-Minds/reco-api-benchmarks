#!/bin/bash
# `sh run.sh XMinds` for example will run all experiments on XMinds API

api="$1"
api="`echo "$api" | tr '[:upper:]' '[:lower:]'`"
case "$api" in
  amazon|google|recombee|xminds|abacus)
    ;;
	*)
	  echo "API=$api not accepted"
	  exit
	  ;;
esac

echo "==============================="
echo ""
echo " - Experiments on '$api' API about to run."
echo ""

# for expfile in api_parameters trivial random medium_pure_clusters medium_hierarchical_clusters big_pure_clusters big_hierarchical_clusters
list_expes="trivial2 medium_unary medium_unary_bis medium_binary medium_decr_pure_embed medium_decr_pure_embed_deep medium_clusters_layers big_unary"
echo " - List of expes to be run (or skipped if already done):"
echo "     $list_expes             "
echo ""
echo "==============================="

for expfile in $list_expes
do
  echo "----------------------------------------"
  echo "----  experiment_${expfile}.yml"
  echo "----------------------------------------"
  python . "experiments/configs/${expfile}.yml" $api
done