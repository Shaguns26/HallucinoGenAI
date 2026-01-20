// Create the Context Menu Item
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "checkHallucination",
    title: "😵‍💫 Check for Hallucinations",
    contexts: ["selection"]
  });
});

// Handle Click
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "checkHallucination") {
    const selectedText = info.selectionText;

    // For this MVP, we assume the "Premise" is the first half
    // and "Hypothesis" is the second half of the selection.
    // In V2, you'd allow users to paste a source URL.
    const splitIndex = Math.floor(selectedText.length / 2);
    const premise = selectedText.substring(0, splitIndex);
    const hypothesis = selectedText.substring(splitIndex);

    // Call API
    fetch("http://127.0.0.1:8000/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ premise: premise, hypothesis: hypothesis })
    })
    .then(response => response.json())
    .then(data => {
      // Alert the user (Simple MVP)
      chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: (status, score) => {
          alert(`Analysis Complete:\nStatus: ${status}\nRisk Score: ${(score * 100).toFixed(1)}%`);
        },
        args: [data.status, data.hallucination_score]
      });
    })
    .catch(error => console.log("Error:", error));
  }
});