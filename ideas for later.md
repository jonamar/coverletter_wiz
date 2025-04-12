#ideas for later

For roles and industries, we can enhance the tagging process by leveraging your CV

also we should clean up all the deprecated files. 

also the mermaid diagram's new line syntax isnt being parsed by the mermaid viewer it just shows up as plain text '/n' 

in legends we cant vote things down so there shouldnt be the option to vote both down

make the default llm for the job feature gemma3:12b

There are a few UI/CLI refinements that aren't fully completed:

Progress Indicators: The CLI doesn't show progress for long-running operations (like analyzing multiple jobs or generating cover letters with larger models)
Command History/Resumption: No ability to resume interrupted operations or track command history
Interactive Selection: The CLI could be enhanced with interactive selection for models and jobs rather than requiring IDs
Color and Formatting: Terminal output lacks color highlighting and formatting that would make results easier to scan


the fundamental flow was 

intake cover letters -> rate content -> analyze jobs -> match content -> generate cover letters/report can you see if this is all working correctly? ie that all the functionality is migrated from the old scripts and also that there are tests for each of the core components. 