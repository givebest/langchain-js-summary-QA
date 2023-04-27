import { useCallback, useState, useEffect } from "react";
import { Typography, Spin, Divider, Button, Space } from "antd";
import { OpenAI } from "langchain/llms/openai";
import { loadSummarizationChain } from "langchain/chains";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { PromptTemplate } from "langchain/prompts";
import { loadQAStuffChain } from "langchain/chains";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { TokenTextSplitter } from "langchain/text_splitter";
import { RetrievalQAChain, loadQARefineChain } from "langchain/chains";
import { text, textShort, qaPrompt } from "./utils";
import "./App.css";

const { Title, Paragraph } = Typography;
const openAIApiKey = import.meta.env.VITE_OPENAI_API_KEY;
let vectorStoreData: MemoryVectorStore;

console.log("openAIApiKey", openAIApiKey);

const model = new OpenAI({
  openAIApiKey,
  temperature: 0.5,
  // modelName: "gpt-3.5-turbo",
  // streaming: true,
});

function App() {
  const [summary, setSummary] = useState("");
  const [title, setTitle] = useState("");
  const [loading, setLoading] = useState(false);

  const onQA = useCallback(async () => {
    setLoading(true);
    const textSplitter = new TokenTextSplitter({
      encodingName: "gpt2",
      chunkSize: 4000,
      chunkOverlap: 0,
    });

    console.log("qaPrompt", qaPrompt);

    const docs = await textSplitter.createDocuments([text]);
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
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1700,
    });
    const docs = await textSplitter.createDocuments([text]);

    const prompt = new PromptTemplate({
      template: `Summarize this content into a bulleted list of the most important information and write in "${lang}" language:
        "{text}"
        CONCISE SUMMARY:`,
      inputVariables: ["text"],
    });

    const chain = loadSummarizationChain(model, {
      prompt,
      combineMapPrompt: prompt,
      combinePrompt: prompt,
    });

    const res = await chain.call({
      input_documents: docs,
      max_tokens: 800,
    });

    setLoading(false);
    setTitle("Summary");
    setSummary(res.text || res.choices[0].text);
  }, []);

  useEffect(() => {
    if (loading) {
      setTitle("");
      setSummary("");
    }
  }, [loading]);

  return (
    <>
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
        <Paragraph>{summary}</Paragraph>
      </Typography>
    </>
  );
}

export default App;
