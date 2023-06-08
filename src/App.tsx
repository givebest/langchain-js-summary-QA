import { useCallback, useState, useEffect } from "react";
import { Typography, Spin, Divider, Button, Space, Input } from "antd";
import ReactMarkdown from "react-markdown";
import { OpenAI } from "langchain/llms/openai";
import { loadSummarizationChain } from "langchain/chains";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { PromptTemplate } from "langchain/prompts";
import { loadQAStuffChain } from "langchain/chains";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { TokenTextSplitter } from "langchain/text_splitter";
import { RetrievalQAChain, loadQARefineChain } from "langchain/chains";
import {
  text,
  textShort,
  qaPrompt,
  textCn,
  textEn,
  mergePrompt,
} from "./utils";
import "./App.css";

const { Title, Paragraph } = Typography;
const { TextArea } = Input;
const openAIApiKey = import.meta.env.VITE_OPENAI_API_KEY;
let vectorStoreData: MemoryVectorStore;

let answerList: string[] = [];

// console.log("openAIApiKey", openAIApiKey);

const model = new OpenAI({
  openAIApiKey,
  temperature: 0,
  maxTokens: 800,
  // modelName: "text-embedding-ada-002",
  modelName: "text-davinci-002",
  // streaming: true,

  callbacks: [
    {
      async handleLLMNewToken(token: string) {
        console.log("token", String(token));
      },
      async handleLLMError(error: string) {
        console.log("error", error);
      },
      handleLLMEnd(res) {
        const text =
          res?.generations &&
          res?.generations[0] &&
          res?.generations[0][0]?.text;
        console.log("end", text);
        answerList.push(text);
      },
    },
  ],
});

function App() {
  const [summary, setSummary] = useState("");
  const [title, setTitle] = useState("");
  const [loading, setLoading] = useState(false);
  const [customPrompt, setCustomPrompt] = useState("");

  const onChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setCustomPrompt(e.target.value);
  }, []);

  const getMergeAnswer = useCallback(async () => {
    const response = await fetch(`https://api.openai.com/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${openAIApiKey}`,
      },
      body: JSON.stringify({
        temperature: 0,
        max_tokens: 800,
        ...{
          // model: "text-davinci-003",
          model: "gpt-3.5-turbo",
          stream: false,
          messages: [
            {
              content: mergePrompt({
                content: answerList.join(""),
                lang: "zh-hans",
                prompt: customPrompt,
              }),
              role: "user",
            },
          ],
        },
      }),
    });

    const res = await response.json();

    setLoading(false);
    const answer =
      res?.text ||
      res?.output_text ||
      (res?.choices &&
        (res?.choices[0]?.text || res?.choices[0]?.message?.content));

    console.log("answer", answer);

    setTitle("Summary");
    setSummary(answer);
    answerList = [];
  }, []);

  const onQA = useCallback(async () => {
    setLoading(true);
    const textSplitter = new TokenTextSplitter({
      encodingName: "gpt2",
      chunkSize: 4000,
      chunkOverlap: 0,
    });

    console.log("qaPrompt", qaPrompt);

    const docs = await textSplitter.createDocuments([textShort]);
    const vectorStore = await MemoryVectorStore.fromDocuments(
      docs,
      new OpenAIEmbeddings({
        openAIApiKey: openAIApiKey,
      })
    );

    vectorStoreData = vectorStore;

    const chain = loadQAStuffChain(model, {
      prompt: qaPrompt,
    });
    const res = await chain.call({
      input_documents: docs,
      question: "内容介绍了什么？",
    });

    console.log("onQA", res);

    setLoading(false);
    setTitle("Q&A");
    setSummary(res.text || res.output_text || res.choices[0].text);
  }, []);

  const onQaAgain = useCallback(async () => {
    console.log("vectorStoreData", vectorStoreData);
    setLoading(true);
    const chain = new RetrievalQAChain({
      // combineDocumentsChain: loadQARefineChain(model),
      combineDocumentsChain: loadQAStuffChain(model, {
        prompt: qaPrompt,
      }),
      retriever: vectorStoreData?.asRetriever(),
    });

    const res = await chain.call({
      query: "里面有哪儿些动物?",
    });

    console.log("onQaAgain", res);
    setLoading(false);
    setTitle("Q&A Again");
    setSummary(res.text || res.output_text || res.choices[0].text);
  }, []);

  const onSummary = useCallback(async () => {
    setLoading(true);
    const lang = "zh-hans";
    const prompt = new PromptTemplate({
      template: `Write a concise summary of the following in the language of the original text:
    ---
    "{text}"
    ---
   CONCISE SUMMARY:`,
      inputVariables: ["text"],
    });

    if (vectorStoreData) {
      const chain = new RetrievalQAChain({
        combineDocumentsChain: loadSummarizationChain(model, {
          type: "map_reduce",
          prompt,
          combineMapPrompt: prompt,
          combinePrompt: prompt,
        }),
        retriever: vectorStoreData?.asRetriever(),
      });

      await chain.call({
        query: customPrompt,
      });

      await getMergeAnswer;
      return;
    }

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
    });
    const docs = await textSplitter.createDocuments([textCn]);
    const vectorStore = await MemoryVectorStore.fromDocuments(
      docs,
      new OpenAIEmbeddings({
        openAIApiKey: openAIApiKey,
      })
    );

    vectorStoreData = vectorStore;

    const chain = loadSummarizationChain(model, {
      type: "map_reduce",
      prompt,
      combineMapPrompt: prompt,
      combinePrompt: prompt,
    });

    await chain.call({
      input_documents: docs,
      max_tokens: 800,
    });

    // setLoading(false);

    // const content = res.text || res.choices[0].text;
    // console.log("summary content", content, answerList);

    getMergeAnswer();
  }, [getMergeAnswer]);

  useEffect(() => {
    if (loading) {
      setTitle("");
      setSummary("");
    }
  }, [loading]);

  return (
    <>
      <TextArea
        rows={4}
        onChange={onChange}
        value={customPrompt}
        placeholder="prompt"
      />
      <Divider />
      <Space wrap>
        <Button type="primary" onClick={onSummary}>
          Summary
        </Button>

        <Button type="primary" onClick={onQA}>
          Q&A
        </Button>

        <Button type="primary" onClick={onQaAgain}>
          Q&A again
        </Button>
      </Space>

      <Divider />
      <Typography>
        <Spin spinning={loading} />
        <Title>{title}</Title>
        <Paragraph className="summary">
          <ReactMarkdown children={summary} />
        </Paragraph>
      </Typography>
    </>
  );
}

export default App;
