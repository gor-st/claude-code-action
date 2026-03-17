import * as core from "@actions/core";
import { readFile, writeFile } from "fs/promises";
import { query } from "@anthropic-ai/claude-agent-sdk";
import type {
  SDKMessage,
  SDKResultMessage,
} from "@anthropic-ai/claude-agent-sdk";
import type { ParsedSdkOptions } from "./parse-sdk-options";

const EXECUTION_FILE = `${process.env.RUNNER_TEMP}/claude-execution-output.json`;

/**
 * Sanitizes SDK output to match CLI sanitization behavior
 */
function sanitizeSdkOutput(
  message: SDKMessage,
  showFullOutput: boolean,
): string | null {
  if (showFullOutput) {
    return JSON.stringify(message, null, 2);
  }

  // System initialization - safe to show
  if (message.type === "system" && message.subtype === "init") {
    return JSON.stringify(
      {
        type: "system",
        subtype: "init",
        message: "Claude Code initialized",
        model: "model" in message ? message.model : "unknown",
      },
      null,
      2,
    );
  }

  // Result messages - show sanitized summary
  if (message.type === "result") {
    const resultMsg = message as SDKResultMessage;
    return JSON.stringify(
      {
        type: "result",
        subtype: resultMsg.subtype,
        is_error: resultMsg.is_error,
        duration_ms: resultMsg.duration_ms,
        num_turns: resultMsg.num_turns,
        total_cost_usd: resultMsg.total_cost_usd,
        permission_denials: resultMsg.permission_denials,
      },
      null,
      2,
    );
  }

  // Suppress other message types in non-full-output mode
  return null;
}

/**
 * Run Claude using the Agent SDK
 */
export async function runClaudeWithSdk(
  promptPath: string,
  { sdkOptions, showFullOutput, hasJsonSchema }: ParsedSdkOptions,
): Promise<void> {
  const prompt = await readFile(promptPath, "utf-8");

  if (!showFullOutput) {
    console.log(
      "Running Claude Code via SDK (full output hidden for security)...",
    );
    console.log(
      "Rerun in debug mode or enable `show_full_output: true` in your workflow file for full output.",
    );
  }

  console.log(`Running Claude with prompt from file: ${promptPath}`);
  // Log SDK options without env (which could contain sensitive data)
  const { env, ...optionsToLog } = sdkOptions;
  console.log("SDK options:", JSON.stringify(optionsToLog, null, 2));

  const messages: SDKMessage[] = [];
  let resultMessage: SDKResultMessage | undefined;

  try {
    for await (const message of query({ prompt, options: sdkOptions })) {
      messages.push(message);

      const sanitized = sanitizeSdkOutput(message, showFullOutput);
      if (sanitized) {
        console.log(sanitized);
      }

      if (message.type === "result") {
        resultMessage = message as SDKResultMessage;
      }
    }
  } catch (error) {
    console.error("SDK execution error:", error);
    core.setOutput("conclusion", "failure");
    process.exit(1);
  }

  // Write execution file
  try {
    await writeFile(EXECUTION_FILE, JSON.stringify(messages, null, 2));
    console.log(`Log saved to ${EXECUTION_FILE}`);
    core.setOutput("execution_file", EXECUTION_FILE);
  } catch (error) {
    core.warning(`Failed to write execution file: ${error}`);
  }

  if (!resultMessage) {
    core.setOutput("conclusion", "failure");
    core.error("No result message received from Claude");
    process.exit(1);
  }

  const isSuccess = resultMessage.subtype === "success";
  core.setOutput("conclusion", isSuccess ? "success" : "failure");

  // Handle structured output - background tasks (subagents) that complete after
  // the main turn can cause the structured_output to be lost in two ways:
  // 1. A follow-up result message without structured_output overwrites the real one
  // 2. The StructuredOutput tool is called in an intermediate turn, and subsequent
  //    background-task turns produce a final result that never gets the field
  if (hasJsonSchema) {
    let structuredOutput: object | undefined;

    // Strategy 1: Search result messages (most recent first) for structured_output
    const structuredResult = [...messages]
      .reverse()
      .find(
        (m) =>
          m.type === "result" &&
          "structured_output" in m &&
          !!(m as SDKResultMessage).structured_output,
      );

    if (
      structuredResult &&
      "structured_output" in structuredResult &&
      structuredResult.structured_output
    ) {
      structuredOutput = structuredResult.structured_output as object;
    }

    // Strategy 2: Fall back to extracting from StructuredOutput tool_use calls
    // in assistant messages. This handles cases where the tool was called in an
    // intermediate turn and background tasks caused additional turns afterward.
    if (!structuredOutput) {
      const toolUseMsg = [...messages].reverse().find((m) => {
        if (m.type !== "assistant" || !("message" in m)) return false;
        const msg = m as {
          message?: { content?: Array<{ type: string; name?: string }> };
        };
        return msg.message?.content?.some(
          (c) => c.type === "tool_use" && c.name === "StructuredOutput",
        );
      });

      if (toolUseMsg && "message" in toolUseMsg) {
        const msg = toolUseMsg as {
          message: {
            content: Array<{
              type: string;
              name?: string;
              input?: Record<string, unknown>;
            }>;
          };
        };
        const toolUse = msg.message.content.find(
          (c) => c.type === "tool_use" && c.name === "StructuredOutput",
        );
        if (toolUse?.input && typeof toolUse.input === "object") {
          structuredOutput = toolUse.input;
          core.info(
            "Recovered structured_output from StructuredOutput tool call (background task race)",
          );
        }
      }
    }

    if (structuredOutput) {
      const structuredOutputJson = JSON.stringify(structuredOutput);
      core.setOutput("structured_output", structuredOutputJson);
      core.info(
        `Set structured_output with ${Object.keys(structuredOutput).length} field(s)`,
      );
    } else {
      core.setFailed(
        `--json-schema was provided but Claude did not return structured_output. Result subtype: ${resultMessage.subtype}`,
      );
      core.setOutput("conclusion", "failure");
      process.exit(1);
    }
  }

  if (!isSuccess) {
    if ("errors" in resultMessage && resultMessage.errors) {
      core.error(`Execution failed: ${resultMessage.errors.join(", ")}`);
    }
    process.exit(1);
  }
}
