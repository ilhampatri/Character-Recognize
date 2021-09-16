from skimage import feature
class HOG:
    def __init__(self, orientations = 18 , pixelsPerCell = (10,10) ,
                cellsPerBlock = (2,2), transform =True,blocknorm="L2-Hys"):
        self.orientations = orientations
        self.pixelsPerCell = pixelsPerCell
        self.cellsPerBlock = cellsPerBlock
        self.transform = transform
        self.blocknorm = blocknorm
        
    def describe(self,image):
        hist =  feature.hog(image,orientations = self.orientations,
                            pixels_per_cell = self.pixelsPerCell,
                            cells_per_block = self.cellsPerBlock,
                            transform_sqrt = self.transform,
                            block_norm = self.blocknorm)
            
        return hist
 

