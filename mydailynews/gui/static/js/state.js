export const state = {
  app: null,
  reports: [],
  currentReport: null,
  config: null,
  configDraft: null,
  userMemoryDraft: null,
  memory: null,
  storyIndexDraft: null,
  learned: null,
  learnedDraft: null,
  runs: [],
  currentRun: null,
  memoryFilters: {
    storySearch: "",
    status: "all",
    feedbackAction: "all",
    topicSource: "",
    dateBrief: "",
  },
  view: "reports",
  reportType: "all",
  reportSort: "date_desc",
  reportGroupsOpen: {},
  contentCollapsed: false,
  browserCollapsed: true,
};

export const feedbackLabels = {
  too_repetitive: "Too repetitive",
  not_relevant: "Not relevant",
  not_interested_in_topic: "Not interested",
  more_like_this: "More like this",
};

export const monthNames = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

export function defaultStoryIndex() {
  return { schema_version: 1, stories: [] };
}
