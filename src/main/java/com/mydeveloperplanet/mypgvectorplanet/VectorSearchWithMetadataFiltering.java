package com.mydeveloperplanet.mypgvectorplanet;

import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocuments;
import static dev.langchain4j.store.embedding.filter.MetadataFilterBuilder.metadataKey;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.utility.DockerImageName;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.pgvector.PgVectorEmbeddingStore;

public class VectorSearchWithMetadataFiltering {

    public static void main(String[] args) throws JsonProcessingException {

        DockerImageName dockerImageName = DockerImageName.parse("pgvector/pgvector:pg16");
        try (PostgreSQLContainer<?> postgreSQLContainer = new PostgreSQLContainer<>(dockerImageName)) {
            postgreSQLContainer.start();

            EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

            EmbeddingStore<TextSegment> embeddingStore = PgVectorEmbeddingStore.builder()
                    .host(postgreSQLContainer.getHost())
                    .port(postgreSQLContainer.getFirstMappedPort())
                    .database(postgreSQLContainer.getDatabaseName())
                    .user(postgreSQLContainer.getUsername())
                    .password(postgreSQLContainer.getPassword())
                    .table("test")
                    .dimension(embeddingModel.dimension())
                    .build();

            embedMarkdown(embeddingModel, embeddingStore);

            invokeSearch(embeddingModel, embeddingStore, "what is the highest chart position of the album \"tracks\" in canada?");

            postgreSQLContainer.stop();
        }
    }

    private static void embedMarkdown(EmbeddingModel embeddingModel, EmbeddingStore<TextSegment> embeddingStore) throws JsonProcessingException {
        List<Document> documents = loadDocuments(toPath("markdown-files"));

        for (Document document : documents) {

            // Split the document line by line
            String[] splittedDocument = document.text().split("\n");

            // split the header on | and remove the first item (the line starts with | and the first item is therefore empty)
            String[] tempSplittedHeader = splittedDocument[0].split("\\|");
            String[] splittedHeader = Arrays.copyOfRange(tempSplittedHeader,1, tempSplittedHeader.length);

            // Preserve only the rows containing data, the first two rows contain the header
            String[] dataOnly = Arrays.copyOfRange(splittedDocument, 2, splittedDocument.length);

            for (String documentLine : dataOnly) {
                // split a data row on | and remove the first item (the line starts with | and the first item is therefore empty)
                String[] tempSplittedDocumentLine = documentLine.split("\\|");
                String[] splittedDocumentLine = Arrays.copyOfRange(tempSplittedDocumentLine, 1, tempSplittedDocumentLine.length);

                ObjectMapper mapper = new ObjectMapper();
                ObjectNode jsonObject = mapper.createObjectNode();

                for (int i = 0; i < splittedHeader.length; i++) {
                    jsonObject.put(splittedHeader[i].strip(), splittedDocumentLine[i].strip());
                }

                String jsonString = mapper.writeValueAsString(jsonObject);

                TextSegment segment = TextSegment.from(jsonString, document.metadata());
                Embedding embedding = embeddingModel.embed(segment).content();
                embeddingStore.add(embedding, segment);

            }

        }

    }

    private static void invokeSearch(EmbeddingModel embeddingModel, EmbeddingStore<TextSegment> embeddingStore, String question) {

        Embedding queryEmbedding = embeddingModel.embed(question).content();

        EmbeddingSearchRequest embeddingSearchRequest = EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .filter(metadataKey("file_name").isEqualTo("bruce_springsteen_discography_compilation_albums.md"))
                .maxResults(5)
                .build();

        List<EmbeddingMatch<TextSegment>> relevant = embeddingStore.search(embeddingSearchRequest).matches();

        StringBuilder inputData = new StringBuilder();
        for (EmbeddingMatch<TextSegment> match : relevant) {
            inputData.append(match.embedded().text()).append("\n");
        }

        System.out.println("Question: " + question);
        for (EmbeddingMatch<TextSegment> match : relevant) {
            System.out.println(match.score());
            System.out.println(match.embedded().text());
            System.out.println(match.embedded().metadata());
        }
        System.out.println();

    }

    private static String createPrompt(String question, String inputData) {
        return "Answer the following question: " + question + "\n" +
                "Use the following data to answer the question: " + inputData;
    }

    private static Path toPath(String fileName) {
        try {
            URL fileUrl = VectorSearchWithMetadataFiltering.class.getClassLoader().getResource(fileName);
            return Paths.get(fileUrl.toURI());
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

}
