const question = document.querySelector("#question");
const ask = document.querySelector("#ask");
const example = document.querySelector("#example");
const status = document.querySelector("#status");
const result = document.querySelector("#result");

const escapeHtml = (value) => value.replace(/[&<>"]/g, (character) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"})[character]);
const renderText = (value) => escapeHtml(value).replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>").replace(/\n/g, "<br>");
const list = (items, tag) => "<" + tag + ">" + items.map((item) => "<li>" + item + "</li>").join("") + "</" + tag + ">";

async function loadExamples() {
  try {
    const response = await fetch("/examples");
    const data = await response.json();
    if (!response.ok || !Array.isArray(data.examples)) throw new Error("Example generation failed");
    example.replaceChildren(new Option("Choose an example…", ""));
    data.examples.forEach((item) => example.add(new Option(item.label, item.question)));
    example.disabled = false;
  } catch (error) {
    example.replaceChildren(new Option("Examples are temporarily unavailable", ""));
    example.disabled = true;
  }
}

example.addEventListener("change", () => {
  if (!example.value) return;
  question.value = example.value;
  question.focus();
});

loadExamples();

ask.addEventListener("click", async () => {
  const value = question.value.trim();
  if (value.length < 3) { status.textContent = "Please enter a research question."; return; }
  ask.disabled = true;
  result.hidden = true;
  status.textContent = "Planning, retrieving evidence, and running the grounding review…";
  try {
    const response = await fetch("/ask", {method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({question:value})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Request failed");
    document.querySelector("#answer").innerHTML = renderText(data.answer);
    document.querySelector("#mode").textContent = data.mode === "llm" ? "LLM-assisted" : "Local fallback";
    document.querySelector("#plan").innerHTML = "<strong>Plan</strong>" + list(data.plan.map((item) => escapeHtml(item)), "ol");
    document.querySelector("#tools").innerHTML = "<strong>Tools chosen</strong>" + list(data.tools_used.map((tool) => "<code>" + escapeHtml(tool.tool) + "</code> — " + escapeHtml(tool.purpose) + " (" + tool.result_count + " results, " + tool.elapsed_ms + " ms; " + escapeHtml(tool.input_summary) + ")"), "ul");
    const revision = data.review.revised ? " · revised once" : "";
    document.querySelector("#review").innerHTML = "<strong>Critic: " + escapeHtml(data.review.status) + revision + "</strong>" + list(data.review.notes.map((note) => escapeHtml(note)), "ul");
    document.querySelector("#citations").innerHTML = data.citations.map((citation) => "<details><summary>" + escapeHtml(citation.source) + " · " + escapeHtml(citation.heading) + " · lines " + escapeHtml(citation.lines) + "</summary><p>" + escapeHtml(citation.excerpt) + "</p></details>").join("");
    result.hidden = false;
    status.textContent = "";
  } catch (error) { status.textContent = "Unable to complete research: " + error.message; }
  finally { ask.disabled = false; }
});

