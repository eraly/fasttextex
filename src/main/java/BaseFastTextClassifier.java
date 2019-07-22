import com.opencsv.CSVReader;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang.ArrayUtils;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.FileLabeledSentenceProvider;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

@Slf4j

/*
author: farizrahman4u
 */
public abstract class BaseFastTextClassifier {

    @Getter
    protected int numClasses;
    @Getter
    protected int inputLength;

    @Setter
    protected ComputationGraph model = null;

    @Getter
    private FastText fastText;

    @Getter
    protected List<String> labels;

    private CnnSentenceDataSetIterator dummyIterator;


    private LabeledSentenceProvider sentenceProvider;
    private LabeledSentenceProvider testSentenceProvider;

    private Map<String, Integer> counts;

    public BaseFastTextClassifier(){
        this(2, 20);
    }

    public BaseFastTextClassifier(int numClasses, int inputLength){
        this.numClasses = numClasses;
        this.inputLength = inputLength;
        log.info("loading fasttext...");
        String fastTextPath  = System.getenv("FASTTEXT_PATH");
        if (fastTextPath == null){
            throw new RuntimeException("Environment variable `FASTTEXT_PATH` is not set! Set it to the location of fasttext embeddings (Download from https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz)");
        }
        File fastTextFile = new File(fastTextPath);
        if (!((fastTextFile.exists() & fastTextFile.isFile()))){
            throw new RuntimeException("File not found: " + fastTextPath + ". Download fasttext embeddings from https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz");
        }
        this.fastText = new FastText(fastTextFile);
        log.info("loading complete.");
        this.dummyIterator = new CnnSentenceDataSetIterator.Builder(CnnSentenceDataSetIterator.Format.CNN1D)
                .sentenceProvider(new CollectionLabeledSentenceProvider(Arrays.asList("a", "b"), Arrays.asList("a", "b")))
                .wordVectors(fastText)
                .maxSentenceLength(inputLength)
                .build();
    }

    protected  ComputationGraphConfiguration getModelConfig(){
            throw new RuntimeException("Unsupported!");
    }

    protected ComputationGraph getModel(){
        if (model == null){
            model = new ComputationGraph(getModelConfig());
        }
        return model;
    }

    protected DataSetIterator getTrainDataSetIterator(int batchSize){
            return new CnnSentenceDataSetIterator.Builder(CnnSentenceDataSetIterator.Format.CNN1D)
                    .sentenceProvider(sentenceProvider)
                    .wordVectors(fastText)
                    .maxSentenceLength(inputLength)
                    .useNormalizedWordVectors(false)
                    .maxSentenceLength(batchSize)
                    .build();

    }
    protected DataSetIterator getTestDataSetIterator(int batchSize){

            return new CnnSentenceDataSetIterator.Builder(CnnSentenceDataSetIterator.Format.CNN1D)
                    .sentenceProvider(testSentenceProvider)
                    .wordVectors(fastText)
                    .maxSentenceLength(inputLength)
                    .useNormalizedWordVectors(false)
                    .maxSentenceLength(batchSize)
                    .build();

    }

    public void setTrainDataFile(File csvFile){
        setSentenceProvider(csvFile, false);

    }
    public void setTrainData(Map<String, List<File>> files){
        setSentenceProvider(files, false);
    }

    private void setSentenceProvider(Map<String, List<File>> files, boolean test){
        if (test) {
            testSentenceProvider = new FileLabeledSentenceProvider(files);
        }
        else {
            labels = Arrays.asList(files.keySet().toArray(new String[files.size()]));
            Collections.sort(labels);
            if (labels.size() > numClasses){
                throw new RuntimeException("labels.size() > numClasses");
            }
            counts = new HashMap<>();
            for (Map.Entry<String, List<File>> entry : files.entrySet()) {
                counts.put(entry.getKey(), entry.getValue().size());
            }
            sentenceProvider = new FileLabeledSentenceProvider(files);
        }

    }

    private static String[] parseUtternaceAndLabel(String[] line){
        if (line.length < 2){
            throw new RuntimeException("Expected 2 comma separated values per line, received - " + Arrays.asList(line));
        }
        String label = line[line.length -1];
        String utterance = "";
        for (int i=0; i < line.length  -1; i++){
            utterance += line[i];
        }
        return new String[]{utterance, label};
    }
    private void setSentenceProvider(File file, boolean test){
        try{
            Reader reader = Files.newBufferedReader(Paths.get(file.getAbsolutePath()));
            CSVReader csvReader = new CSVReader(reader);
            List<String> utterances = new ArrayList<>();
            List<String> labels = new ArrayList<>();
            String[] line = csvReader.readNext();
            Set<String> labelsSet = new HashSet<>();
            counts = new HashMap<>();
            String utterance;
            String label;
            while(line != null){
                try{
                utterance = line[0];
                label = line[1];
                if (utterance.charAt(0) == '\"' && utterance.charAt(utterance.length() - 1) == '\"'){
                    utterance = utterance.substring(1, utterance.length() - 1);
                }
                if (label.charAt(0) == '\"' && label.charAt(label.length() - 1) == '\"'){
                    label = label.substring(1, label.length() - 1);
                }
                if (utterance.isEmpty() || label.isEmpty()){
                    throw new RuntimeException("Empty row: " + line);
                }
                } catch(Exception e){
                    line = csvReader.readNext();
                    continue;
                }

                utterances.add(utterance);
                labels.add(label);
                labelsSet.add(label);
                Integer count = counts.get(label);
                if (count == null){
                    counts.put(label, 1);
                }
                else{
                    counts.put(label, count + 1);
                }
                line = csvReader.readNext();

            }
            if (test){
                testSentenceProvider = new CollectionLabeledSentenceProvider(utterances, labels);
            }
            else{
                this.labels = Arrays.asList(labelsSet.toArray(new String[labelsSet.size()]));
                Collections.sort(this.labels);
                if (labelsSet.size() > numClasses){
                    throw new RuntimeException("labels.size() > numClasses. Received labels: " + labelsSet);
                }
                sentenceProvider = new CollectionLabeledSentenceProvider(utterances, labels);

            }

        }catch (IOException e){
            throw new RuntimeException("Error reading csv file: " + file.getAbsolutePath());
        }
    }

    public void setTrainDataFolders(Map<String, File> folders){
        Map<String, List<File>> files = new HashMap<String, List<File>>();
        for(Map.Entry<String, File> entry: folders.entrySet()){
            if (!entry.getValue().isDirectory()){
                throw new RuntimeException("Not a directory: " + entry.getValue().getAbsolutePath());
            }
            File[] fileArr = entry.getValue().listFiles();
            if (fileArr == null){
                throw new RuntimeException("Empty directory: " + entry.getValue().getAbsolutePath());
            }
            files.put(entry.getKey(), Arrays.asList(fileArr));

        }
        setSentenceProvider(files, false);
    }

    public void setTrainDataFolder(File file){
        Map<String, File> folders = new HashMap<String, File>();
        File[] subFolders = file.listFiles();

        if (subFolders == null){
            throw new RuntimeException("Not a directory: " + file.getAbsolutePath());
        }
        log.info("Checking subfolders...");
        for (File folder: subFolders){
            if (folder.isDirectory()){
                log.info(folder.getName());
                folders.put(folder.getName(), folder);
            }
        }
        setTrainDataFolders(folders);
    }

    public void setTestData(Map<String, List<File>> files){
        setSentenceProvider(files, true);
    }

    public void setTestDataFile(File csvFile){
        setSentenceProvider(csvFile, true);
    }

    public void setTestDataFolders(Map<String, File> folders){
        Map<String, List<File>> files = new HashMap<String, List<File>>();
        for(Map.Entry<String, File> entry: folders.entrySet()){
            if (!entry.getValue().isDirectory()){
                throw new RuntimeException("Not a directory: " + entry.getValue().getAbsolutePath());
            }
            File[] fileArr = entry.getValue().listFiles();
            if (fileArr == null){
                throw new RuntimeException("Empty directory: " + entry.getValue().getAbsolutePath());
            }
            files.put(entry.getKey(), Arrays.asList(fileArr));

        }
        setTestData(files);
    }

    public void setTestDataFolder(File file){
        Map<String, File> folders = new HashMap<String, File>();
        File[] subFolders = file.listFiles();

        if (subFolders == null){
            throw new RuntimeException("Not a directory: " + file.getAbsolutePath());
        }

        for (File folder: subFolders){
            if (folder.isDirectory()){
                folders.put(folder.getName(), folder);
            }
        }
        setTestDataFolders(folders);
    }


    public void setListeners(BaseTrainingListener... listeners){
        this.model.setListeners(listeners);
    }

    public void fit(int batchSize, int numEpochs){
        if (model == null){
            log.info("Building model...");
            model = getModel();
            model.init();
            log.info("Done.");
        }
        log.info("Creating iterator...");
        DataSetIterator iter = getTrainDataSetIterator(batchSize);
        log.info("Done.");
        this.model.setListeners(new ScoreIterationListener(10));
        log.info("fit..");
        this.model.fit(iter, numEpochs);
    }


    public INDArray predict(String input){
        INDArray arr = this.model.output(dummyIterator.loadSingleSentence(input))[0];
        return arr.get(NDArrayIndex.point(arr.shape()[0] - 1));
    }

    public String predictLabel(String input){
        INDArray probs = predict(input);
        log.info(ArrayUtils.toString(probs.shape()));
        if (labels == null){
            return ((Integer)Nd4j.argMax(probs).getInt()).toString();
        }
        return labels.get(Nd4j.argMax(probs).getInt());
    }

    protected INDArray getClassWeights(){
        String[] labels = getLabels().toArray(new String[0]);
        float[] nums = new float[labels.length];
        int sum = 0;
        for (int i=0; i < labels.length; i++){
            int j = counts.get(labels[i]);
            sum += j;
            nums[i] = j;
        }
        for (int i=0; i < labels.length; i++){
            nums[i] = sum  / labels.length * nums[i];
        }
        log.info(Arrays.toString(nums));
        return Nd4j.create(nums, new long[]{nums.length});

    }


}
