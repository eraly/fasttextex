import org.nd4j.linalg.io.ClassPathResource;


/*
author: farizrahman4u
 */
public class FastTextExample {

    public static void main(String[] args) throws Exception{

        int batchSize = 128;
        int nEpochs = 1;
        //Nd4j.create(new double[]{1, 2,3,4}, new long[]{4});
        BaseFastTextClassifier nnClassifier= new FastTextCNN(7, 20);
        // train from folder:
        //nn.setTrainDataFolder(new File("...."));
        // train from single file:
        nnClassifier.setTrainDataFile(new ClassPathResource("fasttextexample/data.csv").getFile());
        nnClassifier.fit(batchSize,nEpochs);
        System.out.println(nnClassifier.predictLabel("book a flight"));

    }
}
