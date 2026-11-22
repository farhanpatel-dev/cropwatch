// Plain-language labels and explanations for the class names in docs/api.md.
export const CLASS_INFO = {
  nutrient_deficiency: {
    label: "Nutrient deficiency",
    what: "Plants short of nutrients such as nitrogen; they look pale or yellow from above.",
  },
  drydown: {
    label: "Drydown",
    what: "Plants drying out, from drought or because the crop is ripening.",
  },
  weed_cluster: {
    label: "Weed cluster",
    what: "Patches where weeds grow among the crop.",
  },
};

export const label = (name) => CLASS_INFO[name]?.label || name.replace(/_/g, " ");
