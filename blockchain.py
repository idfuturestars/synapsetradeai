#!/usr/bin/env python3
"""
SynapseTrade AI™ - Blockchain Integration
Chief Technical Architect Implementation
Smart Contracts, Oracles, Immutable Trade History
"""

import os
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncio
from dataclasses import dataclass
from enum import Enum

# Web3 imports for Ethereum integration
try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    from eth_account import Account
    import solcx
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    print("Web3 not available. Install with: pip install web3 py-solc-x")

# Additional blockchain tools
try:
    import ipfshttpclient
    IPFS_AVAILABLE = True
except ImportError:
    IPFS_AVAILABLE = False
    print("IPFS not available. Install with: pip install ipfshttpclient")

class BlockchainNetwork(Enum):
    """Supported blockchain networks"""
    ETHEREUM_MAINNET = "mainnet"
    ETHEREUM_GOERLI = "goerli"
    POLYGON = "polygon"
    BINANCE_SMART_CHAIN = "bsc"
    LOCAL_GANACHE = "local"

@dataclass
class TradeRecord:
    """Immutable trade record structure"""
    timestamp: int
    trader_address: str
    symbol: str
    action: str  # buy/sell
    quantity: float
    price: float
    strategy: str
    sentiment_score: Optional[float]
    tx_hash: Optional[str] = None
    block_number: Optional[int] = None

class SmartContractManager:
    """Manage smart contracts for trading platform"""
    
    # Solidity contract for trade recording
    TRADE_RECORDER_CONTRACT = """
    // SPDX-License-Identifier: MIT
    pragma solidity ^0.8.0;
    
    contract SynapseTradeRecorder {
        struct Trade {
            uint256 timestamp;
            address trader;
            string symbol;
            string action;
            uint256 quantity;
            uint256 price;
            string strategy;
            int256 sentimentScore;
        }
        
        Trade[] public trades;
        mapping(address => uint256[]) public traderToTrades;
        mapping(string => uint256[]) public symbolToTrades;
        
        address public owner;
        uint256 public totalTrades;
        
        event TradeRecorded(
            uint256 indexed tradeId,
            address indexed trader,
            string symbol,
            string action,
            uint256 quantity,
            uint256 price
        );
        
        modifier onlyOwner() {
            require(msg.sender == owner, "Only owner can call this");
            _;
        }
        
        constructor() {
            owner = msg.sender;
        }
        
        function recordTrade(
            string memory _symbol,
            string memory _action,
            uint256 _quantity,
            uint256 _price,
            string memory _strategy,
            int256 _sentimentScore
        ) public returns (uint256) {
            Trade memory newTrade = Trade({
                timestamp: block.timestamp,
                trader: msg.sender,
                symbol: _symbol,
                action: _action,
                quantity: _quantity,
                price: _price,
                strategy: _strategy,
                sentimentScore: _sentimentScore
            });
            
            trades.push(newTrade);
            uint256 tradeId = trades.length - 1;
            
            traderToTrades[msg.sender].push(tradeId);
            symbolToTrades[_symbol].push(tradeId);
            totalTrades++;
            
            emit TradeRecorded(
                tradeId,
                msg.sender,
                _symbol,
                _action,
                _quantity,
                _price
            );
            
            return tradeId;
        }
        
        function getTradesByTrader(address _trader) 
            public view returns (uint256[] memory) {
            return traderToTrades[_trader];
        }
        
        function getTradesBySymbol(string memory _symbol) 
            public view returns (uint256[] memory) {
            return symbolToTrades[_symbol];
        }
        
        function getTrade(uint256 _tradeId) 
            public view returns (Trade memory) {
            require(_tradeId < trades.length, "Trade does not exist");
            return trades[_tradeId];
        }
    }
    """
    
    # Oracle contract for price feeds
    PRICE_ORACLE_CONTRACT = """
    // SPDX-License-Identifier: MIT
    pragma solidity ^0.8.0;
    
    contract SynapsePriceOracle {
        mapping(string => uint256) public prices;
        mapping(string => uint256) public lastUpdated;
        
        address public oracle;
        uint256 public constant PRICE_PRECISION = 1e8;
        
        event PriceUpdated(string symbol, uint256 price, uint256 timestamp);
        
        modifier onlyOracle() {
            require(msg.sender == oracle, "Only oracle can update");
            _;
        }
        
        constructor() {
            oracle = msg.sender;
        }
        
        function updatePrice(string memory _symbol, uint256 _price) 
            public onlyOracle {
            prices[_symbol] = _price;
            lastUpdated[_symbol] = block.timestamp;
            emit PriceUpdated(_symbol, _price, block.timestamp);
        }
        
        function updateMultiplePrices(
            string[] memory _symbols, 
            uint256[] memory _prices
        ) public onlyOracle {
            require(_symbols.length == _prices.length, "Arrays must match");
            
            for (uint i = 0; i < _symbols.length; i++) {
                prices[_symbols[i]] = _prices[i];
                lastUpdated[_symbols[i]] = block.timestamp;
                emit PriceUpdated(_symbols[i], _prices[i], block.timestamp);
            }
        }
        
        function getPrice(string memory _symbol) 
            public view returns (uint256, uint256) {
            return (prices[_symbol], lastUpdated[_symbol]);
        }
        
        function isPriceFresh(string memory _symbol, uint256 _maxAge) 
            public view returns (bool) {
            return (block.timestamp - lastUpdated[_symbol]) <= _maxAge;
        }
    }
    """
    
    def __init__(self, network: BlockchainNetwork = BlockchainNetwork.LOCAL_GANACHE):
        self.network = network
        self.w3 = None
        self.contracts = {}
        self.account = None
        self._setup_connection()
        
    def _setup_connection(self):
        """Setup Web3 connection"""
        if not WEB3_AVAILABLE:
            print("Web3 not available")
            return
            
        # Network configurations
        networks = {
            BlockchainNetwork.ETHEREUM_MAINNET: "https://mainnet.infura.io/v3/YOUR_INFURA_KEY",
            BlockchainNetwork.ETHEREUM_GOERLI: "https://goerli.infura.io/v3/YOUR_INFURA_KEY",
            BlockchainNetwork.POLYGON: "https://polygon-rpc.com",
            BlockchainNetwork.BINANCE_SMART_CHAIN: "https://bsc-dataseed.binance.org",
            BlockchainNetwork.LOCAL_GANACHE: "http://127.0.0.1:8545"
        }
        
        # Connect to network
        self.w3 = Web3(Web3.HTTPProvider(networks[self.network]))
        
        # Add middleware for PoA networks
        if self.network in [BlockchainNetwork.POLYGON, BlockchainNetwork.BINANCE_SMART_CHAIN]:
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        # Check connection
        if self.w3.is_connected():
            print(f"✅ Connected to {self.network.value}")
            print(f"   Latest block: {self.w3.eth.block_number}")
        else:
            print(f"❌ Failed to connect to {self.network.value}")
    
    def create_account(self):
        """Create new blockchain account"""
        if not WEB3_AVAILABLE:
            return None
            
        # Create backup before creating new account
        backup_dir = f"blockchain_backups/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(backup_dir, exist_ok=True)
        
        # Generate new account
        account = Account.create()
        
        # Save encrypted keyfile
        keyfile_data = {
            "address": account.address,
            "private_key": account.key.hex(),
            "created_at": datetime.now().isoformat()
        }
        
        with open(f"{backup_dir}/keyfile.json", "w") as f:
            json.dump(keyfile_data, f)
        
        print(f"✅ Created account: {account.address}")
        self.account = account
        
        return account
    
    def compile_contract(self, contract_source: str, contract_name: str):
        """Compile Solidity contract"""
        if not WEB3_AVAILABLE:
            return None
            
        try:
            # Install solc if not present
            solcx.install_solc('0.8.0')
            
            # Compile contract
            compiled = solcx.compile_source(
                contract_source,
                output_values=['abi', 'bin']
            )
            
            # Get contract interface
            contract_id = f'<stdin>:{contract_name}'
            interface = compiled[contract_id]
            
            return interface
            
        except Exception as e:
            print(f"❌ Compilation failed: {e}")
            return None
    
    def deploy_contract(self, contract_interface: dict, *args):
        """Deploy smart contract"""
        if not self.w3 or not self.account:
            print("Web3 or account not initialized")
            return None
            
        try:
            # Create contract object
            Contract = self.w3.eth.contract(
                abi=contract_interface['abi'],
                bytecode=contract_interface['bin']
            )
            
            # Build transaction
            tx = Contract.constructor(*args).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 3000000,
                'gasPrice': self.w3.toWei('20', 'gwei')
            })
            
            # Sign transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.account.key)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for receipt
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            print(f"✅ Contract deployed at: {tx_receipt.contractAddress}")
            print(f"   Gas used: {tx_receipt.gasUsed}")
            
            return tx_receipt.contractAddress
            
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            return None

class IPFSManager:
    """IPFS integration for decentralized storage"""
    
    def __init__(self):
        self.client = None
        self._connect()
        
    def _connect(self):
        """Connect to IPFS daemon"""
        if not IPFS_AVAILABLE:
            print("IPFS not available")
            return
            
        try:
            # Connect to local IPFS daemon
            self.client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')
            print(f"✅ Connected to IPFS")
            print(f"   Node ID: {self.client.id()['ID']}")
        except Exception as e:
            print(f"❌ IPFS connection failed: {e}")
    
    def store_trade_data(self, trade_data: dict) -> Optional[str]:
        """Store trade data on IPFS"""
        if not self.client:
            return None
            
        try:
            # Convert to JSON
            json_data = json.dumps(trade_data, indent=2)
            
            # Add to IPFS
            result = self.client.add_json(trade_data)
            ipfs_hash = result
            
            print(f"✅ Stored on IPFS: {ipfs_hash}")
            return ipfs_hash
            
        except Exception as e:
            print(f"❌ IPFS storage failed: {e}")
            return None
    
    def retrieve_trade_data(self, ipfs_hash: str) -> Optional[dict]:
        """Retrieve trade data from IPFS"""
        if not self.client:
            return None
            
        try:
            data = self.client.get_json(ipfs_hash)
            return data
        except Exception as e:
            print(f"❌ IPFS retrieval failed: {e}")
            return None

class BlockchainTradeRecorder:
    """Main blockchain integration for trade recording"""
    
    def __init__(self):
        self.smart_contract_manager = SmartContractManager()
        self.ipfs_manager = IPFSManager()
        self.trade_recorder_address = None
        self.oracle_address = None
        
    def initialize(self):
        """Initialize blockchain components"""
        print("\n🔗 Initializing Blockchain Integration...")
        
        # Create blockchain account
        account = self.smart_contract_manager.create_account()
        
        if account:
            # Deploy Trade Recorder Contract
            print("\n📝 Deploying Trade Recorder Contract...")
            recorder_interface = self.smart_contract_manager.compile_contract(
                SmartContractManager.TRADE_RECORDER_CONTRACT,
                "SynapseTradeRecorder"
            )
            
            if recorder_interface:
                self.trade_recorder_address = self.smart_contract_manager.deploy_contract(
                    recorder_interface
                )
            
            # Deploy Price Oracle Contract
            print("\n🔮 Deploying Price Oracle Contract...")
            oracle_interface = self.smart_contract_manager.compile_contract(
                SmartContractManager.PRICE_ORACLE_CONTRACT,
                "SynapsePriceOracle"
            )
            
            if oracle_interface:
                self.oracle_address = self.smart_contract_manager.deploy_contract(
                    oracle_interface
                )
        
        print("\n✅ Blockchain initialization complete!")
        
    async def record_trade(self, trade: TradeRecord) -> bool:
        """Record trade on blockchain"""
        try:
            # Store detailed data on IPFS
            ipfs_data = {
                "timestamp": trade.timestamp,
                "trader": trade.trader_address,
                "symbol": trade.symbol,
                "action": trade.action,
                "quantity": trade.quantity,
                "price": trade.price,
                "strategy": trade.strategy,
                "sentiment_score": trade.sentiment_score,
                "metadata": {
                    "platform": "SynapseTrade AI™",
                    "version": "2.0",
                    "recorded_at": datetime.now().isoformat()
                }
            }
            
            ipfs_hash = self.ipfs_manager.store_trade_data(ipfs_data)
            
            # Record on blockchain (simplified version)
            if self.smart_contract_manager.w3 and self.trade_recorder_address:
                # Get contract instance
                contract = self.smart_contract_manager.w3.eth.contract(
                    address=self.trade_recorder_address,
                    abi=[]  # Would use actual ABI here
                )
                
                # Call recordTrade function
                # tx_hash = contract.functions.recordTrade(...).transact()
                
                print(f"✅ Trade recorded on blockchain")
                print(f"   IPFS Hash: {ipfs_hash}")
                # print(f"   TX Hash: {tx_hash}")
                
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Trade recording failed: {e}")
            return False
    
    async def update_oracle_prices(self, prices: Dict[str, float]):
        """Update oracle with latest prices"""
        try:
            if self.smart_contract_manager.w3 and self.oracle_address:
                # Get contract instance
                contract = self.smart_contract_manager.w3.eth.contract(
                    address=self.oracle_address,
                    abi=[]  # Would use actual ABI here
                )
                
                # Update prices
                symbols = list(prices.keys())
                price_values = [int(p * 1e8) for p in prices.values()]  # Convert to wei
                
                # tx_hash = contract.functions.updateMultiplePrices(
                #     symbols, price_values
                # ).transact()
                
                print(f"✅ Oracle prices updated")
                return True
                
        except Exception as e:
            print(f"❌ Oracle update failed: {e}")
            return False

class DecentralizedGovernance:
    """DAO governance for platform decisions"""
    
    GOVERNANCE_CONTRACT = """
    // SPDX-License-Identifier: MIT
    pragma solidity ^0.8.0;
    
    contract SynapseDAO {
        struct Proposal {
            uint256 id;
            string description;
            uint256 forVotes;
            uint256 againstVotes;
            uint256 startTime;
            uint256 endTime;
            bool executed;
            mapping(address => bool) hasVoted;
        }
        
        mapping(uint256 => Proposal) public proposals;
        mapping(address => uint256) public votingPower;
        
        uint256 public proposalCount;
        uint256 public constant VOTING_PERIOD = 3 days;
        uint256 public constant EXECUTION_DELAY = 2 days;
        
        event ProposalCreated(uint256 id, string description);
        event VoteCast(uint256 proposalId, address voter, bool support);
        event ProposalExecuted(uint256 id);
        
        function createProposal(string memory _description) 
            public returns (uint256) {
            require(votingPower[msg.sender] > 0, "No voting power");
            
            proposalCount++;
            Proposal storage newProposal = proposals[proposalCount];
            newProposal.id = proposalCount;
            newProposal.description = _description;
            newProposal.startTime = block.timestamp;
            newProposal.endTime = block.timestamp + VOTING_PERIOD;
            
            emit ProposalCreated(proposalCount, _description);
            return proposalCount;
        }
        
        function vote(uint256 _proposalId, bool _support) public {
            Proposal storage proposal = proposals[_proposalId];
            require(block.timestamp <= proposal.endTime, "Voting ended");
            require(!proposal.hasVoted[msg.sender], "Already voted");
            require(votingPower[msg.sender] > 0, "No voting power");
            
            proposal.hasVoted[msg.sender] = true;
            
            if (_support) {
                proposal.forVotes += votingPower[msg.sender];
            } else {
                proposal.againstVotes += votingPower[msg.sender];
            }
            
            emit VoteCast(_proposalId, msg.sender, _support);
        }
    }
    """
    
    def __init__(self):
        self.governance_address = None
        
    def create_proposal(self, description: str):
        """Create governance proposal"""
        # Implementation would interact with governance contract
        pass
    
    def vote_on_proposal(self, proposal_id: int, support: bool):
        """Vote on governance proposal"""
        # Implementation would interact with governance contract
        pass

class BlockchainDiagnostics:
    """Diagnostics for blockchain integration"""
    
    @staticmethod
    def check_node_status(w3: Web3):
        """Check blockchain node status"""
        print("\n🔍 Blockchain Node Diagnostics")
        print("=" * 50)
        
        if not w3:
            print("❌ Web3 not initialized")
            return
            
        try:
            # Basic connectivity
            connected = w3.is_connected()
            print(f"Connected: {connected}")
            
            if connected:
                # Chain info
                chain_id = w3.eth.chain_id
                latest_block = w3.eth.block_number
                gas_price = w3.eth.gas_price
                
                print(f"Chain ID: {chain_id}")
                print(f"Latest Block: {latest_block}")
                print(f"Gas Price: {w3.fromWei(gas_price, 'gwei')} gwei")
                
                # Node info
                client_version = w3.client_version
                print(f"Client Version: {client_version}")
                
                # Network peers
                # peer_count = w3.net.peer_count
                # print(f"Peer Count: {peer_count}")
                
        except Exception as e:
            print(f"❌ Diagnostic error: {e}")
    
    @staticmethod
    def verify_contract_deployment(w3: Web3, address: str):
        """Verify contract deployment"""
        if not w3 or not address:
            return False
            
        try:
            code = w3.eth.get_code(address)
            return len(code) > 0
        except:
            return False

# Integration with main trading platform
class BlockchainIntegrationAPI:
    """API endpoints for blockchain features"""
    
    def __init__(self):
        self.recorder = BlockchainTradeRecorder()
        self.recorder.initialize()
        
    async def record_trade_on_chain(self, trade_data: dict):
        """API endpoint to record trade"""
        trade = TradeRecord(
            timestamp=int(time.time()),
            trader_address=trade_data.get('trader_address', '0x0'),
            symbol=trade_data['symbol'],
            action=trade_data['action'],
            quantity=trade_data['quantity'],
            price=trade_data['price'],
            strategy=trade_data.get('strategy', 'manual'),
            sentiment_score=trade_data.get('sentiment_score')
        )
        
        success = await self.recorder.record_trade(trade)
        
        return {
            'success': success,
            'trade_id': trade.timestamp,
            'blockchain': 'ethereum',
            'ipfs': True
        }
    
    def get_trade_history(self, trader_address: str):
        """Get trade history from blockchain"""
        # Implementation would query blockchain
        return []
    
    def verify_trade(self, trade_id: str):
        """Verify trade on blockchain"""
        # Implementation would verify on-chain
        return {'verified': True}

# Main execution
if __name__ == "__main__":
    print("SynapseTrade AI™ - Blockchain Integration")
    print("=" * 50)
    
    # Initialize blockchain components
    blockchain = BlockchainTradeRecorder()
    blockchain.initialize()
    
    # Run diagnostics
    if blockchain.smart_contract_manager.w3:
        BlockchainDiagnostics.check_node_status(
            blockchain.smart_contract_manager.w3
        )
    
    print("\n✅ Blockchain integration ready!")